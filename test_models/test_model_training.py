"""
Isolated Test Model Training Script
Comparative testing of 5 hazard detection architectures
Each model uses its own architecture-specific loss function
Does NOT interfere with src/ modules
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import random


# Import from src (data loading only)
from src.data.dataloader import get_dataloaders
from src.train import HazardTrainer
from src.config import DATA_DIR, PROJECT_ROOT

# Import test models
from test_models.models.unet_model import UNetModel
from test_models.models.deeplab_model import DeepLabV3PlusModel
from test_models.models.segformer_model import SegFormerModel
from test_models.models.hazard_cnn_model import HazardCNNModel

# Import test loss functions (architecture-specific)
from test_models.losses.deeplab_focal_loss import DeepLabFocalLoss
from test_models.losses.segformer_ce_loss import SegFormerCrossEntropyLoss
from test_models.losses.hazard_cnn_focal_loss import HazardCNNFocalLoss

def set_seed(seed_value=42):
    """Set seed for reproducibility to ensure different models have different initializations."""
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def run_model_test():
    """
    Runs comparative test of all 5 models with architecture-specific loss functions
    """
    print("\n" + "="*80)
    print("HAZARDLOC: ISOLATED MODEL COMPARATIVE TESTING")
    print("="*80)
    print("Testing 4 architectures, standardized to classification with Focal/CE Loss")
    print("\nArchitecture → Loss Function Assignment:")
    print(" 1. U-Net (ResNet18) → Focal Loss")
    print(" 2. DeepLabV3+ (ResNet18) → Focal Loss")
    print(" 3. SegFormer (MiT-B0) → CrossEntropy Loss")
    print(" 4. ResNet-18 (HazardCNN) → Focal Loss")
    print("="*80)

        # --- Device Setup ---
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        print(f"\n✓ TPU device found: {xm.xla_real_devices([str(device)])}")
    except ImportError:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")

    # --- Configuration for testing ---
    test_config = {
        'epochs': 1,
        'batch_size': 16,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'patience': 1,
        'use_bf16': False, # Explicitly disable mixed precision for test models
        'num_workers': 0,
        'loss_alpha': 1.0,
        'loss_beta': 0.5
    }

    print(f"\nTest Configuration: {test_config}")

    # --- Load Data ---
    print(f"\nLoading data from: {DATA_DIR}/processed")
    processed_data_dir = Path(f"{DATA_DIR}/processed")

    if not processed_data_dir.exists() or not any(processed_data_dir.iterdir()):
        print(f"✗ ERROR: Processed data not found at '{processed_data_dir}'")
        print("  Please run the data preprocessing script first.")
        return

    data_loaders = get_dataloaders(
        str(processed_data_dir),
        batch_size=test_config['batch_size'],
        num_workers=test_config['num_workers']
    )

    num_classes = len(data_loaders['class_names'])
    print(f"✓ Dataset loaded with {num_classes} classes: {data_loaders['class_names']}")

    # --- Define a dictionary of model classes and their loss functions ---
    # We will instantiate them INSIDE the loop to ensure a fresh model for each test.
    model_definitions = {
        "U-Net (ResNet18)": (
            UNetModel,
            HazardCNNFocalLoss() # Reusing a standard focal loss for 2D output
        ),
        "DeepLabV3+ (ResNet18)": (
            DeepLabV3PlusModel,
            DeepLabFocalLoss()
        ),
        "SegFormer (MiT-B0)": (
           SegFormerModel,
           SegFormerCrossEntropyLoss() # Retaining CE loss as it's standard for SegFormer
        ),
        "ResNet-18 (HazardCNN)": (
            HazardCNNModel,
            HazardCNNFocalLoss()
        ),
    }

    results = []

    # --- Iterate and Test Each Model ---
    for i, (model_name, (ModelClass, criterion)) in enumerate(model_definitions.items()):
        print(f"\n{'='*80}")
        print(f"TESTING: {model_name}")
        print(f"Loss Function: {criterion.__class__.__name__}")
        print(f"{'='*80}")

        # --- CRITICAL FIX: Set a different seed for each model test run ---
        # This ensures each model gets a unique random initialization and data shuffle.
        set_seed(42 + i)

        # --- CRITICAL FIX: Create data subsets INSIDE the loop ---
        # This ensures that each model test uses a new, independent random subset of data.
        subset_fraction = 0.1
        original_train_dataset = data_loaders['train'].dataset
        original_val_dataset = data_loaders['val'].dataset

        train_indices = torch.randperm(len(original_train_dataset))[:int(len(original_train_dataset) * subset_fraction)]
        val_indices = torch.randperm(len(original_val_dataset))[:int(len(original_val_dataset) * subset_fraction)]

        train_subset = Subset(original_train_dataset, train_indices)
        val_subset = Subset(original_val_dataset, val_indices)

        train_loader_subset = DataLoader(train_subset, batch_size=test_config['batch_size'], shuffle=True, num_workers=test_config['num_workers'])
        val_loader_subset = DataLoader(val_subset, batch_size=test_config['batch_size'], shuffle=False, num_workers=test_config['num_workers'])
        
        print(f"\nUsing {subset_fraction*100:.0f}% subset for this run:")
        print(f"  Training set: {len(train_subset)} samples")
        print(f"  Validation set: {len(val_subset)} samples")
        subset_dataloaders = {'train': train_loader_subset, 'val': val_loader_subset}

        # --- CRITICAL FIX: Instantiate a fresh model for each test run ---
        # This prevents using the weights from a previously trained model.
        model_instance = ModelClass(num_classes=num_classes)

        # Optimizer
        optimizer = optim.AdamW(
            model_instance.parameters(),
            lr=test_config['learning_rate'],
            weight_decay=test_config['weight_decay']
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=1
        )

        # --- SIMPLIFICATION: Use the standard HazardTrainer for all models ---
        trainer = HazardTrainer(model_instance, device, test_config)
        
        # Train
        history = trainer.train(
            subset_dataloaders['train'],
            subset_dataloaders['val'],
            optimizer,
            criterion,
            scheduler,
            save_model=False,
            epochs_to_run=test_config['epochs']
        )

        # --- IMPROVEMENT: Get final metrics directly from training history ---
        # This avoids running a redundant validation step after training is complete.
        # The `train` method already validates at the end of each epoch.
        final_val_metrics = {
            'f1': history['val_f1'][-1],
            'recall': history['val_recall'][-1],
            'macro_recall': history['val_macro_recall'][-1],
            'accuracy': history['val_acc'][-1],
            'iou': history['val_iou'][-1],
            'precision': history['val_precision'][-1],
            'roc_auc': history['val_roc_auc'][-1]
        }

        # Store results
        results.append({
            "Model": model_name,
            "F1 Score": final_val_metrics['f1'],
            "Precision": final_val_metrics['precision'],
            "Recall (Weighted)": final_val_metrics['recall'],
            "Macro Recall": final_val_metrics['macro_recall'],
            "ROC AUC": final_val_metrics['roc_auc'],
            "IoU": final_val_metrics['iou'],
            "Accuracy": final_val_metrics['accuracy'],
            "Loss Function": criterion.__class__.__name__
        })

    # --- Results Summary ---
    results_df = pd.DataFrame(results)
    # Reorder columns for better readability
    column_order = ["Model", "Accuracy", "ROC AUC", "F1 Score", "IoU", "Precision", "Recall (Weighted)", "Macro Recall", "Loss Function"]
    results_df = results_df[column_order]
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    print(results_df.to_string(index=False))

    # --- Visualization ---
    metrics_to_plot = ["Accuracy", "ROC AUC", "F1 Score", "IoU", "Precision", "Macro Recall"]
    df_melted = results_df.melt(id_vars=['Model'], value_vars=metrics_to_plot, var_name='Metric', value_name='Score')

    plt.figure(figsize=(16, 8))
    sns.barplot(x='Model', y='Score', hue='Metric', data=df_melted, palette='viridis')
    plt.title('Model Comparison (Standardized to Classification)', fontsize=16, fontweight='bold')
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('Model Architecture', fontsize=12)
    plt.ylim(0, 1.0)
    plt.xticks(rotation=15, ha='right')
    plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plot_path = Path(PROJECT_ROOT) / 'results' / 'visualizations' / 'test_model_comparison.png'
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=300)
    print(f"\n✓ Comparison chart saved to: {plot_path}")
    plt.show()

    # Save results CSV
    csv_path = Path(PROJECT_ROOT) / 'results' / 'test_model_results.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(csv_path, index=False)
    print(f"✓ Results saved to: {csv_path}")

    print("\n" + "="*80)
    print("✅ MODEL COMPARATIVE TESTING COMPLETE!")
    print("="*80)


if __name__ == '__main__':

    run_model_test()