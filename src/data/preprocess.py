# Preprocessing routines

"""
Data Preprocessing Module for HazardLoc
Handles image preprocessing, augmentation, and train/val/test splitting
"""

import os
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split
import json

class HazardDataPreprocessor:
    """
    Preprocessor for hazard detection images
    """
    
    def __init__(self, raw_dir, processed_dir, target_size=(224, 224)):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.target_size = target_size
        
        # Standard ImageNet normalization
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
    def create_splits(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Split dataset into train/val/test sets
        """
        print("Creating train/val/test splits...")
        
        # Find all images
        image_paths = list(self.raw_dir.glob("**/*.jpg")) + \
                     list(self.raw_dir.glob("**/*.png")) + \
                     list(self.raw_dir.glob("**/*.jpeg"))
        
        # Extract labels from directory structure (if available)
        labels = [img.parent.name for img in image_paths]
        unique_labels = sorted(set(labels))
        
        print(f"Found {len(image_paths)} images")
        print(f"Classes: {unique_labels}")
        
        # Split data
        train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
            image_paths, labels, train_size=train_ratio, stratify=labels, random_state=42
        )
        
        val_imgs, test_imgs, val_labels, test_labels = train_test_split(
            temp_imgs, temp_labels, 
            test_size=test_ratio/(val_ratio + test_ratio),
            stratify=temp_labels, random_state=42
        )
        
        # Save splits
        splits = {
            'train': (train_imgs, train_labels),
            'val': (val_imgs, val_labels),
            'test': (test_imgs, test_labels)
        }
        
        for split_name, (imgs, lbls) in splits.items():
            split_dir = self.processed_dir / split_name
            
            # Create class directories
            for label in unique_labels:
                (split_dir / label).mkdir(parents=True, exist_ok=True)
            
            # Copy images
            for img_path, label in zip(imgs, lbls):
                dest = split_dir / label / img_path.name
                shutil.copy2(img_path, dest)
            
            print(f"  {split_name}: {len(imgs)} images")
        
        # Save split metadata
        metadata = {
            'splits': {
                'train': len(train_imgs),
                'val': len(val_imgs),
                'test': len(test_imgs)
            },
            'classes': unique_labels,
            'target_size': self.target_size
        }
        
        with open(self.processed_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return splits
    
    def preprocess_image(self, image_path, save_path=None):
        """
        Preprocess a single image
        - Resize to target size
        - Convert to RGB
        - Normalize
        """
        img = Image.open(image_path).convert('RGB')
        
        # Resize
        img_resized = img.resize(self.target_size, Image.LANCZOS)
        
        if save_path:
            img_resized.save(save_path)
        
        return img_resized
    
    def compute_dataset_statistics(self):
        """
        Compute mean and std for the dataset
        """
        print("Computing dataset statistics...")
        
        image_paths = list(self.processed_dir.glob("**/*.jpg")) + \
                     list(self.processed_dir.glob("**/*.png"))
        
        # Sample 1000 images for efficiency
        sample_paths = np.random.choice(image_paths, 
                                       min(1000, len(image_paths)), 
                                       replace=False)
        
        pixel_sum = np.zeros(3)
        pixel_sq_sum = np.zeros(3)
        pixel_count = 0
        
        for img_path in sample_paths:
            img = np.array(Image.open(img_path).convert('RGB')) / 255.0
            pixel_sum += img.sum(axis=(0, 1))
            pixel_sq_sum += (img ** 2).sum(axis=(0, 1))
            pixel_count += img.shape[0] * img.shape[1]
        
        mean = pixel_sum / pixel_count
        std = np.sqrt(pixel_sq_sum / pixel_count - mean ** 2)
        
        print(f"Dataset mean: {mean}")
        print(f"Dataset std: {std}")
        
        return mean.tolist(), std.tolist()
    
    def get_transforms(self, augment=False):
        """
        Get preprocessing transforms
        """
        if augment:
            return transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
        else:
            return transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])


def main():
    """
    Main preprocessing workflow
    """
    from src.config import DATA_DIR, PROJECT_ROOT
    
    raw_dir = Path(DATA_DIR) / "raw"
    processed_dir = Path(DATA_DIR) / "processed"
    
    print("="*60)
    print("HAZARDLOC DATA PREPROCESSING")
    print("="*60)
    
    preprocessor = HazardDataPreprocessor(raw_dir, processed_dir)
    
    # Create splits
    preprocessor.create_splits(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    # Compute statistics
    mean, std = preprocessor.compute_dataset_statistics()
    
    print("\n" + "="*60)
    print("Preprocessing complete!")
    print("="*60)
    print(f"Processed data saved to: {processed_dir}")
    print(f"Dataset mean: {mean}")
    print(f"Dataset std: {std}")
    

if __name__ == "__main__":
    main()
