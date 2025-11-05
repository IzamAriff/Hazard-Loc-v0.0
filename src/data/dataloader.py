
"""
Enhanced DataLoader for HazardLoc
Integrates augmentation and efficient data loading
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
from pathlib import Path

class HazardDataset(Dataset):
    """
    Custom Dataset for hazard detection with flexible augmentation
    """

    def __init__(self, root_dir, transform=None, class_names=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        # Load image paths and labels
        self.image_paths = []
        self.labels = []

        # Auto-detect classes if not provided
        if class_names is None:
            class_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
            self.class_names = sorted([d.name for d in class_dirs])
        else:
            self.class_names = class_names

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}

        # Load all images
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue

            for img_ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
                for img_path in class_dir.glob(img_ext):
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])

        print(f"Loaded {len(self.image_paths)} images from {root_dir}")
        print(f"Classes: {self.class_names}")
        print(f"Distribution: {dict(zip(*np.unique(self.labels, return_counts=True)))}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_weights(self):
        """
        Compute class weights for imbalanced datasets
        """
        labels = np.array(self.labels)
        class_counts = np.bincount(labels)
        total = len(labels)
        weights = total / (len(class_counts) * class_counts)
        return torch.FloatTensor(weights)


def get_dataloaders(data_dir, batch_size=32, num_workers=4, augment=True):
    """
    Create train/val/test dataloaders with proper augmentation

    Args:
        data_dir: Path to processed data directory
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        augment: Whether to apply augmentation to training data

    Returns:
        Dictionary with train/val/test dataloaders
    """
    from src.data.augmentation import HazardAugmentation

    data_path = Path(data_dir)
    augmenter = HazardAugmentation()

    # Define transforms
    train_transform = augmenter.get_train_transforms() if augment else augmenter.get_val_transforms()
    val_transform = augmenter.get_val_transforms()
    test_transform = augmenter.get_test_transforms()

    # Create datasets
    train_dataset = HazardDataset(
        data_path / 'train',
        transform=train_transform
    )

    val_dataset = HazardDataset(
        data_path / 'val',
        transform=val_transform,
        class_names=train_dataset.class_names
    )

    test_dataset = HazardDataset(
        data_path / 'test',
        transform=test_transform,
        class_names=train_dataset.class_names
    )

    # Compute class weights for imbalanced data
    class_weights = train_dataset.get_class_weights()

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'class_weights': class_weights,
        'class_names': train_dataset.class_names
    }


if __name__ == '__main__':
    # Test dataloader
    from src.config import DATA_DIR

    loaders = get_dataloaders(DATA_DIR + '/processed', batch_size=16)

    print("\nDataLoader Test:")
    print(f"Train batches: {len(loaders['train'])}")
    print(f"Val batches: {len(loaders['val'])}")
    print(f"Test batches: {len(loaders['test'])}")
    print(f"Class weights: {loaders['class_weights']}")
    print(f"Class names: {loaders['class_names']}")

    # Test one batch
    images, labels = next(iter(loaders['train']))
    print(f"\nBatch shape: {images.shape}")
    print(f"Label shape: {labels.shape}")