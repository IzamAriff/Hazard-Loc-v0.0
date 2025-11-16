"""
Enhanced Training Module for HazardLoc
Includes modern training techniques: early stopping, learning rate scheduling, mixed precision
"""

import torch
import torch.nn as nn
import torch.optim as optim; from torch.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import json
from pathlib import Path
from datetime import datetime

from src.data.dataloader import get_dataloaders
from src.models.hazard_cnn import HazardCNN
from src.utils.metrics import compute_metrics
from src.utils.losses import CompoundLoss # Import CompoundLoss
from src.utils.logger import TrainingLogger
from src.config import DATA_DIR, MODEL_SAVE, PROJECT_ROOT


class HazardTrainer:
    """
    Comprehensive trainer for hazard detection model
    """

    def __init__(self, model, device, config):
        self.model = model.to(device)
        self.device = device
        self.config = config

        # --- MODIFICATION: Check if we are on a TPU device ---
        self.is_tpu = 'xla' in str(self.device)
        if self.is_tpu:
            import torch_xla.core.xla_model as xm
            self.xm = xm

        # BF16 Mixed Precision Setup (as per dissertation Section 3.4.3.3)
        # Determine if mixed precision should be attempted at all
        self.use_mixed_precision = config.get('use_bf16', True) and self.device.type == 'cuda'
        self.dtype = torch.float32 # Default to full precision
        
        if self.use_mixed_precision:
            capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
            if self.is_tpu: # TPUs have native BF16 support
                self.dtype = torch.bfloat16
                print(f"✓ BF16 mixed precision enabled (native TPU support)")
            elif capability[0] >= 8: # Ampere or newer for native BF16 on GPU
                self.dtype = torch.bfloat16
                print(f"✓ GPU (Compute Capability {capability[0]}.{capability[1]}) supports BF16. Using BF16 mixed precision.")
            elif capability[0] >= 7: # Volta or newer for robust FP16 on GPU
                self.dtype = torch.float16
                print(f"✓ GPU (Compute Capability {capability[0]}.{capability[1]}) supports FP16. Using FP16 mixed precision.")
            else:
                # Fallback for older GPUs that lack robust mixed precision support
                self.dtype = torch.float32 # Explicitly fall back to FP32
                self.use_mixed_precision = False
                print(f"⚠ GPU (Compute Capability {capability[0]}.{capability[1]}) has limited mixed precision support. Falling back to FP32 for stability.")
        else:
            self.dtype = torch.float32
            print("Mixed precision disabled. Using FP32.")
        
        # GradScaler only needed for FP16, not BF16
        self.scaler = GradScaler(enabled=(self.dtype == torch.float16))
        
        # Rest of init...
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history = {
            'train_loss': [], 
            'train_acc': [], 
            'train_f1': [], 
            'val_loss': [], 
            'val_acc': [], 
            'val_f1': [],
            'val_precision': [],
            'val_recall': [],
            'val_macro_recall': [],
            'val_roc_auc': []
        }

        # Logger
        self.logger = TrainingLogger(Path(PROJECT_ROOT) / 'results' / 'logs')

    def train_epoch(self, dataloader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []

        pbar = tqdm(dataloader, desc='Training')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            optimizer.zero_grad()

            # BF16/FP16 Mixed Precision Forward Pass
            # --- MODIFICATION: Use 'xla' device type for autocast on TPU ---
            device_type = 'xla' if self.is_tpu else 'cuda'
            with autocast(device_type=device_type, dtype=self.dtype, enabled=self.use_mixed_precision):
                outputs = self.model(images)
                # --- FIX: Unpack the tuple from CompoundLoss ---
                # Check if the criterion returns a tuple (our custom loss)
                loss_output = criterion(outputs, labels)
                if isinstance(loss_output, tuple):
                    loss, loss_components = loss_output
                else:
                    loss = loss_output # Standard loss function
                    loss_components = {}
            
            # Backward pass with proper scaling
            if self.scaler.is_enabled():  # FP16 on GPU
                self.scaler.scale(loss).backward() # type: ignore
            elif self.is_tpu: # TPU
                loss.backward()
                self.xm.optimizer_step(optimizer) # Use XLA optimizer step
            else:  # BF16 or FP32 on GPU/CPU - no scaling needed
                loss.backward()

            # Step scaler for FP16
            if self.scaler.is_enabled():
                self.scaler.step(optimizer)
                self.scaler.update()
            elif not self.is_tpu: # For standard CPU/GPU, step the optimizer manually
                optimizer.step()

            # Statistics
            running_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            _, predicted = outputs.max(1)
            all_probs.extend(torch.softmax(outputs, dim=1).detach().cpu().numpy())
            all_preds.extend(predicted.detach().cpu().numpy())

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
            })

        epoch_loss = running_loss / len(dataloader)
        metrics = compute_metrics(y_true=all_labels, y_pred=all_preds, y_prob=np.array(all_probs))

        return epoch_loss, metrics

    def validate(self, dataloader, criterion):
        """Validate model"""
        self.model.eval()
        running_loss = 0.0
        all_probs = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc='Validation'):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                # --- FIX: Unpack the tuple from CompoundLoss ---
                loss_output = criterion(outputs, labels)
                if isinstance(loss_output, tuple):
                    loss, _ = loss_output
                else:
                    loss = loss_output

                # --- FIX: Ensure predicted labels are collected, not raw logits ---
                _, predicted_labels = outputs.max(1)
                all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
                all_preds.extend(predicted_labels.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                running_loss += loss.item() # Accumulate loss once per batch

        val_loss = running_loss / len(dataloader)
        metrics = compute_metrics(y_true=all_labels, y_pred=all_preds, y_prob=np.array(all_probs))

        return val_loss, metrics

    def train(self, train_loader, val_loader, optimizer, criterion, scheduler=None, save_model=True, epochs_to_run=None):
        """Full training loop"""
        epochs = epochs_to_run if epochs_to_run is not None else self.config.get('epochs', 50)
        patience = self.config.get('patience', 10)

        # --- MODIFICATION: Wrap dataloaders for TPU ---
        if self.is_tpu:
            from torch_xla.distributed.parallel_loader import MpDeviceLoader
            train_loader = MpDeviceLoader(train_loader, self.device)
            val_loader = MpDeviceLoader(val_loader, self.device)

        print(f"\nStarting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{epochs}")
            print('='*60)

            # Train
            train_loss, train_metrics = self.train_epoch(train_loader, optimizer, criterion)

            # Validate
            val_loss, val_metrics = self.validate(val_loader, criterion)

            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%, Train F1: {train_metrics['f1']:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_metrics['accuracy']:.2f}%, Val F1:   {val_metrics['f1']:.4f}")

            # Scheduler step
            if scheduler:
                scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['train_f1'].append(train_metrics['f1'])
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['val_macro_recall'].append(val_metrics['macro_recall'])
            self.history['val_roc_auc'].append(val_metrics.get('roc_auc', 0.0)) # Use .get for safety

            # Early stopping and model saving
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                print(f"✓ Validation loss improved to {self.best_val_loss:.4f}")

                if save_model:
                    # Ensure the directory for saving the model exists
                    Path(MODEL_SAVE).parent.mkdir(parents=True, exist_ok=True)
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'val_metrics': val_metrics,
                    }
                    if self.is_tpu:
                        self.xm.save(checkpoint, MODEL_SAVE)
                    else:
                        torch.save(checkpoint, MODEL_SAVE)
                    print(f"✓ Model saved to: {MODEL_SAVE}")
            else:
                self.patience_counter += 1
                print(f"⚠ No improvement for {self.patience_counter} epochs")

                if self.patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    break

        # --- Ensure a model is always saved at the end of training ---
        # If no improvement was ever made, the model might not have been saved.
        # Save the final state of the model regardless, if saving is enabled.
        if save_model and (not Path(MODEL_SAVE).exists() or self.best_val_loss == float('inf')):
            Path(MODEL_SAVE).parent.mkdir(parents=True, exist_ok=True)
            final_checkpoint = {
                'epoch': epochs - 1, # Or the last actual epoch if early stopped
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            if self.is_tpu:
                self.xm.save(final_checkpoint, MODEL_SAVE)
            else:
                torch.save(final_checkpoint, MODEL_SAVE)
            print(f"\n✓ Final model state saved to: {MODEL_SAVE} (no improvement during training or initial save)")
        # Save training history
        if save_model:
            history_path = Path(MODEL_SAVE).parent / 'training_history.json'
            with open(history_path, 'w') as f:
                json.dump(self.history, f, indent=2)

        print(f"\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        if save_model:
            print(f"Model saved to: {MODEL_SAVE}")

        # --- MODIFICATION: Add Colab download ---
        # Automatically download the model file if running in Google Colab
        try:
            from google.colab import files
            print("\nRunning in Google Colab. Triggering model download...")
            files.download(MODEL_SAVE)
        except ImportError:
            pass  # Not in a Colab environment, do nothing.

        # Close the logger
        self.logger.close()

        return self.history


def main():
    """Main training workflow"""

    # Configuration
    config = {
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'patience': 15,
        'mixed_precision': True,
        'num_workers': 2,
        'loss_alpha': 1.0,  # Focal loss weight
        'loss_beta': 0.5    # Dice loss weight
    }

    print("="*60)
    print("HAZARDLOC MODEL TRAINING")
    print("="*60)

    # Device
    # --- IMPROVEMENT: Centralized TPU/GPU/CPU device detection ---
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        print(f"\n✓ TPU device found: {xm.xla_real_devices([str(device)])[0]}")
    except ImportError:
        print("\n`torch_xla` not found, checking for CUDA/CPU.")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        capability = torch.cuda.get_device_capability(0)
        print(f"Compute Capability: {capability[0]}.{capability[1]}")
        print(f"BF16 Support: {'Yes' if capability[0] >= 8 else 'No (using FP16)'}")

    # Load data
    print(f"\nLoading data from: {DATA_DIR}/processed")
    processed_data_dir = Path(f"{DATA_DIR}/processed")
    if not processed_data_dir.exists() or not any(processed_data_dir.iterdir()):
        print(f"✗ ERROR: Processed data not found at '{processed_data_dir}'")
        print("  Please run the data preprocessing script first to generate train/val/test splits.")
        return

    data = get_dataloaders(
        str(processed_data_dir),
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )

    # Create model
    num_classes = len(data['class_names'])
    model = HazardCNN(num_classes=num_classes)
    print(f"\nModel: HazardCNN with {num_classes} classes")

    # Compound Loss Function (as per dissertation)
    criterion = CompoundLoss(
        alpha_weight=config['loss_alpha'],
        beta_weight=config['loss_beta']
    )
    print(f"Loss: Compound (Focal + Dice) with α={config['loss_alpha']}, β={config['loss_beta']}")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    # Trainer
    trainer = HazardTrainer(model, device, config)

    # Train
    history = trainer.train(
        data['train'],
        data['val'],
        optimizer,
        criterion,
        scheduler
    )

    # Log the final configuration to the logger
    trainer.logger.log_config(config)

    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()