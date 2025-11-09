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

        # BF16 Mixed Precision Setup (as per dissertation Section 3.4.3.3)
        self.use_bf16 = config.get('use_bf16', True) and torch.cuda.is_available()
        
        if self.use_bf16:
            # Check if GPU supports BF16
            if torch.cuda.get_device_capability()[0] >= 8:  # Ampere or newer
                self.dtype = torch.bfloat16
                print("✓ BF16 mixed precision enabled (native GPU support)")
            else:
                print("⚠ GPU does not support BF16, falling back to FP16")
                self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        
        # GradScaler only needed for FP16, not BF16
        self.scaler = GradScaler('cuda') if (self.dtype == torch.float16) else None
        
        # Rest of init...
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

        # Logger
        self.logger = TrainingLogger(Path(PROJECT_ROOT) / 'results' / 'logs')

    def train_epoch(self, dataloader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc='Training')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            optimizer.zero_grad()

            # BF16/FP16 Mixed Precision Forward Pass
            with autocast(device_type='cuda', dtype=self.dtype, enabled=self.use_bf16):
                outputs = self.model(images)
                loss = criterion(outputs, labels)
            
            # Backward pass with proper scaling
            if self.scaler:  # Only for FP16
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:  # BF16 or FP32 - no scaling needed
                loss.backward()
                optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate(self, dataloader, criterion):
        """Validate model"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc='Validation'):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = running_loss / len(dataloader)
        metrics = compute_metrics(all_labels, all_preds)

        return val_loss, metrics

    def train(self, train_loader, val_loader, optimizer, criterion, scheduler=None):
        """Full training loop"""
        epochs = self.config.get('epochs', 50)
        patience = self.config.get('patience', 10)

        print(f"\nStarting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{epochs}")
            print('='*60)

            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)

            # Validate
            val_loss, val_metrics = self.validate(val_loader, criterion)

            # Scheduler step
            if scheduler:
                scheduler.step(val_loss)

            # Log results
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.2f}%")
            print(f"Val Precision: {val_metrics['precision']:.4f} | Val Recall: {val_metrics['recall']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}")

            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_metrics['accuracy'])

            # Early stopping and model saving
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0

                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_metrics': val_metrics,
                }, MODEL_SAVE)
                print(f"✓ Model saved (val_loss improved to {val_loss:.4f})")
            else:
                self.patience_counter += 1
                print(f"⚠ No improvement for {self.patience_counter} epochs")

                if self.patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    break

        # Save training history
        history_path = Path(MODEL_SAVE).parent / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        print(f"\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Model saved to: {MODEL_SAVE}")

        return self.history


def main():
    """Main training workflow"""

    # Configuration
    config = {
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'patience': 5,
        'mixed_precision': True,
        'num_workers': 4,
        'loss_alpha': 1.0,  # Focal loss weight
        'loss_beta': 0.5    # Dice loss weight
    }

    print("="*60)
    print("HAZARDLOC MODEL TRAINING")
    print("="*60)
    print(f"Configuration: {json.dumps(config, indent=2)}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

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
    from losses import CompoundLoss
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

    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()