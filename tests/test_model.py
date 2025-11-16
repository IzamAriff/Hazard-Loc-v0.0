import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import segmentation_models_pytorch as smp

# Import necessary modules from src
from src.data.dataloader import get_dataloaders
from src.utils.losses import CompoundLoss, FocalLoss
from src.models.hazard_cnn import HazardCNN
from src.train import HazardTrainer # We will use the modified HazardTrainer
from src.config import DATA_DIR, PROJECT_ROOT

# --- FIX: Wrapper to adapt segmentation models for classification ---
class SegmentationToClassificationWrapper(nn.Module):
    """
    Wraps a segmentation model to make it suitable for a classification task.
    It applies global average pooling to the output of the segmentation model.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Global average pooling layer to collapse spatial dimensions
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.model(x)  # Output shape: [B, C, H, W]
        x = self.pool(x)   # Output shape: [B, C, 1, 1]
        x = torch.flatten(x, 1) # Output shape: [B, C]
        return x

# --- NEW: Ensemble Model Definition ---
class EnsembleModel(nn.Module):
    """
    A simple ensemble model that averages the outputs (logits) of multiple models.
    """
    def __init__(self, models):
        super().__init__()
        # Use ModuleList to ensure models are properly registered
        self.models = nn.ModuleList(models)

    def forward(self, x):
        # Get outputs from each model
        outputs = [model(x) for model in self.models]
        
        # Stack and average the outputs along a new dimension
        # This averages the logits before the softmax/loss calculation
        stacked_outputs = torch.stack(outputs)
        avg_output = torch.mean(stacked_outputs, dim=0)
        return avg_output

def run_model_test():
    """
    Runs a comparative test for different models with a small training run.
    """
    print("\n" + "="*70)
    print("HAZARDLOC: MODEL COMPARATIVE TESTING")
    print("="*70)

    # --- Device Setup ---
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        print(f"\n✓ TPU device found: {xm.xla_real_devices([str(device)])[0]}")
    except ImportError:
        print("\n`torch_xla` not found, checking for CUDA/CPU.")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")

    # --- Configuration for testing ---
    test_config = {
        'epochs': 3, # Much smaller size for quick testing
        'batch_size': 16, # Smaller batch size for potentially smaller datasets or memory constraints
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'patience': 1, # No early stopping for short runs
        'use_bf16': True,
        'num_workers': 0, # Reduced workers for quick testing, avoid multiprocessing overhead
        'loss_alpha': 1.0,
        'loss_beta': 0.5
    }

    # --- Load Data ---
    print(f"\nLoading data from: {DATA_DIR}/processed")
    processed_data_dir = Path(f"{DATA_DIR}/processed")
    if not processed_data_dir.exists() or not any(processed_data_dir.iterdir()):
        print(f"✗ ERROR: Processed data not found at '{processed_data_dir}'")
        print("  Please run the data preprocessing script first to generate train/val/test splits.")
        return

    data_loaders = get_dataloaders(
        str(processed_data_dir),
        batch_size=test_config['batch_size'],
        num_workers=test_config['num_workers']
    )

    # --- FIX: Get num_classes BEFORE overwriting the data_loaders dictionary ---
    num_classes = len(data_loaders['class_names'])
    print(f"Dataset loaded with {num_classes} classes.")

    # --- MODIFICATION: Create smaller subsets for faster testing ---
    subset_fraction = 0.1  # Use 10% of the data
    print(f"\nUsing a {subset_fraction*100:.0f}% subset of the data for efficient testing.")

    # Get original datasets
    original_train_dataset = data_loaders['train'].dataset
    original_val_dataset = data_loaders['val'].dataset

    # Create random indices for the subset
    train_indices = torch.randperm(len(original_train_dataset))[:int(len(original_train_dataset) * subset_fraction)]
    val_indices = torch.randperm(len(original_val_dataset))[:int(len(original_val_dataset) * subset_fraction)]

    # Create subset datasets
    train_subset = Subset(original_train_dataset, train_indices)
    val_subset = Subset(original_val_dataset, val_indices)

    # Create new dataloaders from the subsets
    train_loader_subset = DataLoader(train_subset, batch_size=test_config['batch_size'], shuffle=True, num_workers=test_config['num_workers'])
    val_loader_subset = DataLoader(val_subset, batch_size=test_config['batch_size'], shuffle=False, num_workers=test_config['num_workers'])

    print(f"  New training set size: {len(train_subset)} samples")
    print(f"  New validation set size: {len(val_subset)} samples")

    # Create a new dictionary for the subset loaders to pass to the trainer
    subset_dataloaders = {'train': train_loader_subset, 'val': val_loader_subset}

    # --- Define Models & Their Loss Functions ---
    # Architecture-specific loss assignment per dissertation methodology

    # 1. U-Net: Segmentation architecture (outputs 4D) -> Compound Loss (Focal + Dice)
    unet_model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
    )
    unet_criterion = CompoundLoss(
        alpha_weight=test_config['loss_alpha'], 
        beta_weight=test_config['loss_beta']
    )

    # 2. DeepLabV3+: Edge detection (wrapped for classification) -> Focal Loss
    deeplab_model = SegmentationToClassificationWrapper(smp.DeepLabV3Plus(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
    ))
    deeplab_criterion = FocalLoss()

    # 3. SegFormer: Transformer-based (wrapped for classification) -> CrossEntropy
    segformer_model = SegmentationToClassificationWrapper(smp.Unet(
        encoder_name="mit_b0",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
    ))
    segformer_criterion = nn.CrossEntropyLoss()

    # 4. ResNet-18 Baseline (classification) -> Focal Loss
    hazard_cnn_model = HazardCNN(num_classes=num_classes)
    hazard_cnn_criterion = FocalLoss()

    # 5. Ensemble (all wrapped for classification) -> Focal Loss
    ensemble_model = EnsembleModel([
        SegmentationToClassificationWrapper(smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
        )),
        deeplab_model,
        segformer_model
    ])
    ensemble_criterion = FocalLoss()

    # Dictionary of models with their designated losses
    models_to_test = {
        "U-Net (ResNet18)": (unet_model, unet_criterion),
        "DeepLabV3+ (ResNet18)": (deeplab_model, deeplab_criterion),
        "SegFormer (MiT-B0)": (segformer_model, segformer_criterion),
        "ResNet-18 (HazardCNN)": (hazard_cnn_model, hazard_cnn_criterion),
        "Ensemble (U-Net+DeepLab+SegFormer)": (ensemble_model, ensemble_criterion),
    }

    results = []

    # --- Iterate and Test Each Model ---
    for model_name, (model_instance, criterion) in models_to_test.items():
        print(f"\n{'='*70}")
        print(f"TESTING MODEL: {model_name}")
        print(f"{'='*70}")

        # Re-initialize optimizer for each model
        optimizer = optim.AdamW(
            model_instance.parameters(),
            lr=test_config['learning_rate'],
            weight_decay=test_config['weight_decay']
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=1
        )

        # Initialize trainer
        trainer = HazardTrainer(model_instance, device, test_config)
        
        # Run training
        history = trainer.train(
            subset_dataloaders['train'],
            subset_dataloaders['val'],
            optimizer,
            criterion,
            scheduler,
            save_model=False,
            epochs_to_run=test_config['epochs']
        )

        # Get final validation metrics
        try:
            final_val_loss, final_val_metrics, final_loss_components = trainer.validate(
                subset_dataloaders['val'], criterion, return_loss_components=True
            )
        except Exception as e:
            print(f"⚠ Warning: Could not get loss components: {e}")
            final_loss_components = {}
        
        # Calculate Dice Coefficient only if available
        dice_coefficient = 1.0 - final_loss_components.get('dice', 0.0) if 'dice' in final_loss_components else None
        
        # Store results
        results.append({
            "Model": model_name,
            "F1 Score": final_val_metrics['f1'],
            "Precision": final_val_metrics['precision'],
            "Recall": final_val_metrics['recall'],
            "Dice Coefficient": dice_coefficient if dice_coefficient is not None else "N/A",
            "Accuracy": final_val_metrics['accuracy'],
            "Loss Function": criterion.__class__.__name__
        })
        
        # Close logger
        trainer.logger.close()


    # --- Process and Visualize Results ---
    results_df = pd.DataFrame(results)
    print("\n" + "="*70)
    print("MODEL COMPARISON RESULTS")
    print("="*70)
    print(results_df.round(4))

    # Create bar chart
    metrics_to_plot = ["F1 Score", "Precision", "Recall", "Dice Coefficient (IOU Proxy)", "Accuracy"]
    
    # Melt the DataFrame for easier plotting with seaborn
    df_melted = results_df.melt(id_vars='Model', value_vars=metrics_to_plot, var_name='Metric', value_name='Score')

    plt.figure(figsize=(14, 8))
    sns.barplot(x='Model', y='Score', hue='Metric', data=df_melted, palette='viridis')
    plt.title('Comparative Performance of Hazard Detection Models (3 Epochs)', fontsize=16)
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('Model Architecture', fontsize=12)
    plt.ylim(0, 1.0) # Scores are typically between 0 and 1
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Save the plot
    plot_path = Path(PROJECT_ROOT) / 'results' / 'visualizations' / 'model_comparison_barchart.png'
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=300)
    print(f"\n✓ Model comparison bar chart saved to: {plot_path}")
    plt.show()

    print("\n" + "="*70)
    print("MODEL COMPARATIVE TESTING COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    run_model_test()