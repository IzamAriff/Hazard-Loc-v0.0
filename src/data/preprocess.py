"""
Handles preprocessing of raw data into a structured format for training.

This script takes the raw data (e.g., from 'data/raw') which contains
class folders ('Positive', 'Negative'), and splits them into 'train' and 'val'
subdirectories within a 'processed' folder.
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

def create_processed_dataset(raw_dir: str, processed_dir: str, val_split: float = 0.2, force: bool = False):
    """
    Splits raw image data into training and validation sets, handling nested directories.

    Args:
        raw_dir (str): Path to the raw data directory containing class folders.
        processed_dir (str): Path to the destination directory for processed data.
        val_split (float): The proportion of the dataset to allocate to the validation set.
        force (bool): If True, overwrites the existing processed directory.
    """
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)

    if not raw_path.exists() or not any(raw_path.iterdir()):
        print(f"✗ ERROR: Raw data directory '{raw_dir}' is empty or does not exist.")
        return False

    if processed_path.exists() and force:
        print(f"Removing existing processed directory: {processed_path}")
        shutil.rmtree(processed_path)
    elif processed_path.exists() and not force:
        print(f"✓ Processed data already exists at '{processed_path}'. Skipping preprocessing.")
        return True

    print(f"Creating processed dataset at: {processed_path}")

    # --- ROBUST LOGIC to handle nested directories like 'Decks/Cracked' ---
    images_by_class = defaultdict(list)
    print("Scanning for images in subdirectories...")

    # Recursively find all .jpg images. This is robust to various folder structures.
    all_images = list(raw_path.glob('**/*.jpg'))
    if not all_images:
        print(f"✗ ERROR: No '.jpg' images found recursively in '{raw_dir}'. Please check the dataset contents.")
        return False

    for img_path in all_images:
        # The immediate parent folder's name is the class (e.g., 'Cracked', 'Non-cracked')
        class_name = img_path.parent.name
        images_by_class[class_name].append(img_path)

    class_names = sorted(list(images_by_class.keys()))
    print(f"Found classes: {class_names}")

    for class_name in class_names:
        # Create train and val directories for each class
        (processed_path / 'train' / class_name).mkdir(parents=True, exist_ok=True)
        (processed_path / 'val' / class_name).mkdir(parents=True, exist_ok=True)

        # Get all image files for the class and shuffle them
        images = images_by_class[class_name]
        random.shuffle(images)

        # Split files into training and validation sets
        split_idx = int(len(images) * val_split)
        val_images = images[:split_idx]
        train_images = images[split_idx:]

        # Copy files to the new 'processed' directories
        for img_path in train_images:
            shutil.copy(img_path, processed_path / 'train' / class_name / img_path.name)
        for img_path in val_images:
            shutil.copy(img_path, processed_path / 'val' / class_name / img_path.name)

        print(f"  - Class '{class_name}': {len(train_images)} train, {len(val_images)} val")

    print("\n✓ Preprocessing complete.")
    return True