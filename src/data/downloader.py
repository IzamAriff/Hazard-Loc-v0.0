"""
Handles downloading and extracting datasets from Kaggle.
"""

import os
import kaggle
from pathlib import Path

def download_from_kaggle(dataset_slug: str, destination: str, force: bool = False):
    """
    Downloads and unzips a dataset from Kaggle into a specified destination directory.

    Args:
        dataset_slug (str): The slug of the Kaggle dataset (e.g., 'user/dataset-name').
        destination (str): The path to the directory where data should be saved.
        force (bool): If True, re-downloads the data even if it already exists.
    """
    dest_path = Path(destination)

    # Check if data exists and if we should skip download
    # A simple check is to see if the destination directory is not empty.
    if dest_path.exists() and any(dest_path.iterdir()) and not force:
        print(f"✓ Dataset already exists at '{destination}'. Skipping download.")
        return True

    print(f"Downloading dataset '{dataset_slug}' to '{destination}'...")

    # Ensure the destination directory exists
    dest_path.mkdir(parents=True, exist_ok=True)

    try:
        # Use the Kaggle API to download and unzip files
        kaggle.api.dataset_download_files(
            dataset_slug,
            path=destination,
            unzip=True,
            force=force  # `force=True` in API overwrites existing files
        )
        print("✓ Download and extraction complete.")
        return True
    except Exception as e:
        print(f"✗ ERROR: Failed to download dataset from Kaggle. Please check your Kaggle API setup and dataset slug.")
        print(f"  Details: {e}")
        print("\n  To set up the Kaggle API, make sure you have a 'kaggle.json' file in your '~/.kaggle/' directory.")
        return False