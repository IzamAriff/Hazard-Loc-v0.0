"""
Specialized Trainer for Segmentation Models in Testing
Handles 4D segmentation outputs [B, C, H, W] for both loss and metrics.
Does NOT interfere with src/train.py
"""

import torch
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from src.utils.metrics import compute_metrics

class SegmentationHazardTrainer:
    """
    A specialized version of HazardTrainer for models that output 4D segmentation masks.
    It correctly calculates image-level metrics from pixel-level predictions.
    """
    def __init__(self, model, device, config):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.use_mixed_precision = config.get('use_bf16', False) # FP32 for stability in tests
        self.dtype = torch.float32
        self.scaler = GradScaler(enabled=False) # Disabled for FP32
        self.best_val_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        self.history = {'train_loss': [], 'train_acc': [], 'train_f1': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
        print("✓ Initialized SegmentationHazardTrainer (for 4D outputs)")

    def train_epoch(self, dataloader, optimizer, criterion):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(dataloader, desc='Training (Seg)')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()

            with autocast(device_type='cuda', dtype=self.dtype, enabled=self.use_mixed_precision):
                outputs = self.model(images) # Expected shape: [B, C, H, W]
                if outputs.dim() != 4:
                    raise ValueError(f"Segmentation trainer expects 4D output, but got {outputs.shape}")
                
                loss_output = criterion(outputs, labels)
                if isinstance(loss_output, tuple):
                    loss, _ = loss_output
                else:
                    loss = loss_output

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # --- METRIC CALCULATION FOR SEGMENTATION OUTPUT ---
            # Convert 4D pixel-wise predictions to 1D image-level predictions
            # An image is a "hazard" (class 1) if any pixel is predicted as class 1.
            _, pixel_preds = outputs.max(1) # [B, H, W]
            predicted_image_level = (pixel_preds == 1).any(dim=-1).any(dim=-1).long() # [B]
            
            all_preds.extend(predicted_image_level.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
            })

        epoch_loss = running_loss / len(dataloader)
        # Compute all metrics at the end of the epoch
        metrics = compute_metrics(all_labels, all_preds)
        
        return epoch_loss, metrics

    def validate(self, dataloader, criterion):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        last_loss_components = {}

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc='Validation (Seg)'):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images) # Expected shape: [B, C, H, W]

                loss_output = criterion(outputs, labels)
                if isinstance(loss_output, tuple):
                    loss, loss_components = loss_output
                    last_loss_components = loss_components
                else:
                    loss = loss_output

                running_loss += loss.item()

                # --- METRIC CALCULATION FOR SEGMENTATION OUTPUT ---
                _, pixel_preds = outputs.max(1)
                predicted_image_level = (pixel_preds == 1).any(dim=-1).any(dim=-1).long()

                all_preds.extend(predicted_image_level.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = running_loss / len(dataloader)
        metrics = compute_metrics(all_labels, all_preds)

        return val_loss, metrics, last_loss_components

    def train(self, train_loader, val_loader, optimizer, criterion, scheduler=None, save_model=True, epochs_to_run=None):
        epochs = epochs_to_run if epochs_to_run is not None else self.config.get('epochs', 1)

        print(f"\nStarting segmentation training for {epochs} epochs...")
        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{epochs}")
            print('='*60)

            train_loss, train_metrics = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_metrics, _ = self.validate(val_loader, criterion)

            if scheduler:
                scheduler.step(val_loss)

            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%, Train F1: {train_metrics['f1']:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_metrics['accuracy']:.2f}%, Val F1:   {val_metrics['f1']:.4f}")

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['train_f1'].append(train_metrics['f1'])
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1'])

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss

        print(f"\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

        return self.history