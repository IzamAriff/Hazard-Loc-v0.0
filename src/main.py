
"""
HazardLoc: Complete Integration Pipeline
Runs end-to-end workflow from 2D detection to 3D localization
"""

import torch
import numpy as np
from pathlib import Path
import importlib

# Import all modules
from src.data.dataloader import get_dataloaders
from src.data.downloader import download_from_kaggle
from src.data.preprocess import create_processed_dataset
from src.utils.losses import CompoundLoss # Import CompoundLoss
from src.models.hazard_cnn import HazardCNN
from src.train import HazardTrainer
from src.utils.colmap_utils import COLMAPAdapter, read_colmap_outputs
# Programmatic import to handle directory name starting with a number
backproject_module = importlib.import_module("src.3d.backproject")
HazardLocalizer = backproject_module.HazardLocalizer
open3d_viz_module = importlib.import_module("src.3d.open3d_viz")
HazardVisualizer = open3d_viz_module.HazardVisualizer
from src.detect import predict_img
from src.config import DATA_DIR, MODEL_SAVE, COLMAP_IMG, COLMAP_OUT, VISUAL_DIR, KAGGLE_DATASET_SLUG


class HazardLocPipeline:
    """
    End-to-end pipeline for HazardLoc system
    """

    def __init__(self):
        # --- MODIFICATION: Add TPU device detection ---
        try:
            import torch_xla.core.xla_model as xm
            self.device = xm.xla_device()
            print(f"✓ TPU device found: {xm.xla_real_devices([str(self.device)])[0]}")
        except ImportError:
            print("torch_xla not found, defaulting to CUDA/CPU.")
            self.device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        
        self.model = None
        self.colmap = None
        self.localizer = None
        self.visualizer = None

    def step0_prepare_data(self, force_download=False):
        """
        Step 0: Download and prepare data from Kaggle
        """
        print("\n" + "="*70)
        print("STEP 0: PREPARING DATASET FROM KAGGLE")
        print("="*70)

        # The raw data will be downloaded here, preserving Kaggle's folder structure.
        # Your dataloader expects data in DATA_DIR/processed, so you might have a
        # separate script to process the raw data into the processed format.
        # For now, we just download it to DATA_DIR/raw.
        raw_data_dir = Path(DATA_DIR) / "raw"

        success = download_from_kaggle(KAGGLE_DATASET_SLUG, str(raw_data_dir), force=force_download)
        if not success:
            print("\n✗ Halting pipeline due to data download failure.")
        return success

    def step0_5_preprocess_data(self, force_preprocess=False):
        """
        Step 0.5: Preprocess raw data into train/val splits.
        """
        print("\n" + "="*70)
        print("STEP 0.5: PREPROCESSING RAW DATA")
        print("="*70)

        raw_data_dir = Path(DATA_DIR) / "raw"
        processed_data_dir = Path(DATA_DIR) / "processed"

        success = create_processed_dataset(str(raw_data_dir), str(processed_data_dir), force=force_preprocess)

        if not success:
            print("\n✗ Halting pipeline due to data preprocessing failure.")

        return success

    def step1_train_detector(self, config=None):
        """
        Step 1: Train 2D hazard detection model
        """
        print("\n" + "="*70)
        print("STEP 1: TRAINING 2D HAZARD DETECTION MODEL")
        print("="*70)

        if config is None:
            config = {
                'epochs': 50,
                'batch_size': 32, # Align with train.py for stability
                'learning_rate': 1e-3,
                'weight_decay': 1e-4, # Align with train.py
                'patience': 10, # Align with train.py
                'use_bf16': True, # Equivalent to mixed_precision: True
                'num_workers': 4, # Align with train.py
                'loss_alpha': 1.0,  # Focal loss weight
                'loss_beta': 0.5    # Dice loss weight
            }

        # --- IMPROVEMENT: Automatic Learning Rate Scaling for TPU ---
        learning_rate = config['learning_rate']
        is_tpu = 'xla' in str(self.device)
        if is_tpu:
            # For TPUs, scale the learning rate by the number of cores.
            # This compensates for the larger effective batch size.
            num_cores = 8  # Default to 8 cores for a standard TPU device
            try:
                # Try the most common modern location first
                import torch_xla.core.xla_model as xm
                num_cores = xm.xrt_world_size()
            except (ImportError, AttributeError):
                try:
                    # Try a previously common location as a fallback
                    from torch_xla.distributed import xla_env
                    num_cores = xla_env.xrt_world_size()
                except (ImportError, AttributeError):
                    print("⚠ WARNING: Could not dynamically determine TPU core count due to torch_xla API changes.")
                    print(f"  Falling back to a default of {num_cores} cores for learning rate scaling.")
            learning_rate *= num_cores
            print(f"✓ TPU detected. Scaling learning rate by {num_cores} cores to: {learning_rate:.1e}")

        # Load data
        data = get_dataloaders(f"{DATA_DIR}/processed", batch_size=config['batch_size'], num_workers=config['num_workers'])

        # Create and train model
        num_classes = len(data['class_names'])
        self.model = HazardCNN(num_classes=num_classes)

        # Compound Loss Function (as per dissertation)
        criterion = CompoundLoss(
            alpha_weight=config['loss_alpha'],
            beta_weight=config['loss_beta']
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=config['weight_decay'] # Use weight_decay from config
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

        trainer = HazardTrainer(self.model, self.device, config)
        history = trainer.train(data['train'], data['val'], optimizer, criterion, scheduler)

        print(f"\n✓ Model trained and saved to: {MODEL_SAVE}")
        return history

    def step2_run_colmap(self):
        """
        Step 2: Run COLMAP 3D reconstruction
        """
        print("\n" + "="*70)
        print("STEP 2: COLMAP 3D RECONSTRUCTION")
        print("="*70)

        self.colmap = COLMAPAdapter(COLMAP_OUT, COLMAP_IMG)
        success = self.colmap.run_full_pipeline()

        if success:
            print(f"\n✓ 3D reconstruction complete. Output: {COLMAP_OUT}")
        else:
            print("\n✗ COLMAP reconstruction failed")

        return success

    def step3_detect_hazards(self, test_images):
        """
        Step 3: Detect hazards in test images
        """
        print("\n" + "="*70)
        print("STEP 3: DETECTING HAZARDS IN IMAGES")
        print("="*70)

        detections = []

        try:
            for img_path in test_images:
                label, confidence = predict_img(img_path)

                if label == 1:  # Hazard detected (assuming 1 = hazard class)
                    # --- IMPROVEMENT ---
                    # Instead of a fixed placeholder, use a bounding box that covers the entire image.
                    # This is a better temporary solution before implementing a real object detector.
                    # We get the image dimensions from the COLMAP camera data later, but for now,
                    # we can read the image to get its size.
                    from PIL import Image
                    with Image.open(img_path) as img:
                        width, height = img.size

                    detections.append({
                        'image': img_path.name,
                        'confidence': confidence,
                        'bbox': [0, 0, width, height]
                    })
                    print(f"  ✓ Hazard detected in {img_path.name} (confidence: {confidence:.2%})")
                else:
                    print(f"  - No hazard in {img_path.name}")
        except FileNotFoundError as e:
            print(f"\n✗ ERROR: {e}")
            print("  Cannot run hazard detection. Halting pipeline.")
            return None

        print(f"\nTotal hazards detected: {len(detections)}")
        return detections

    def step4_localize_3d(self, detections):
        """
        Step 4: Localize hazards in 3D space
        """
        print("\n" + "="*70)
        print("STEP 4: LOCALIZING HAZARDS IN 3D")
        print("="*70)

        # Load COLMAP outputs
        cameras, images, points3D = read_colmap_outputs(COLMAP_OUT, COLMAP_IMG)
        self.localizer = HazardLocalizer(cameras, images, points3D)

        hazard_3d_locations = []

        for detection in detections:
            loc_3d = self.localizer.localize_hazard(
                detection['image'],
                detection['bbox']
            )

            if loc_3d is not None:
                hazard_3d_locations.append(loc_3d)
                print(f"  ✓ {detection['image']}: 3D coords = {loc_3d}")
            else:
                print(f"  ✗ Failed to localize {detection['image']}")

        print(f"\nSuccessfully localized {len(hazard_3d_locations)} hazards in 3D")
        return hazard_3d_locations

    def step5_visualize(self, hazard_locations):
        """
        Step 5: Visualize hazards in 3D point cloud
        """
        print("\n" + "="*70)
        print("STEP 5: VISUALIZING 3D HAZARD MAP")
        print("="*70)

        self.visualizer = HazardVisualizer(COLMAP_OUT, COLMAP_IMG)
        self.visualizer.load_point_cloud()

        for loc in hazard_locations:
            self.visualizer.add_hazard(loc)

        # Save visualization
        output_path = Path(VISUAL_DIR) / "hazard_map_3d.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.visualizer.save_visualization(output_path)

        print(f"\n✓ Visualization saved to: {output_path}")

        # Interactive view
        print("\nLaunching interactive 3D viewer...")
        self.visualizer.visualize()

    def run_complete_pipeline(self, skip_data_prep=False, skip_preprocessing=False, skip_training=False, skip_colmap=False):
        """
        Execute complete HazardLoc pipeline
        """
        print("\n" + "="*70)
        print("HAZARDLOC: COMPLETE PIPELINE EXECUTION")
        print("="*70)

        # Step 0: Prepare data from Kaggle
        if not skip_data_prep:
            if not self.step0_prepare_data():
                return # Stop if data prep fails
        else:
            print("\nSkipping data preparation (assuming data already exists)")

        # Step 0.5: Preprocess data
        if not skip_preprocessing:
            if not self.step0_5_preprocess_data():
                return # Stop if preprocessing fails
        else:
            print("\nSkipping data preprocessing (assuming data is already processed)")

        model_path = Path(MODEL_SAVE)

        # Step 1: Train model (optional skip if already trained)
        if skip_training:
            if not model_path.exists():
                print(f"⚠  WARNING: 'skip_training' is True, but model not found at '{model_path}'.")
                print("           Training is required. Forcing training step...")
                self.step1_train_detector()
            else:
                print(f"\nSkipping training (using existing model at '{model_path}')")
        else:
            print("\nStarting model training...")
            self.step1_train_detector()

        # Step 2: COLMAP reconstruction (optional skip if already done)
        if not skip_colmap:
            success = self.step2_run_colmap()
            if not success:
                print("\nHalting pipeline due to COLMAP failure.")
                return
        else:
            print("\nSkipping COLMAP (using existing reconstruction)")

        # Step 3: Detect hazards
        test_images = list(Path(COLMAP_IMG).glob("*.jpg"))  # Process all images
        detections = self.step3_detect_hazards(test_images)
        if detections is None: # Check if detection step failed
            print("\nHalting pipeline due to error in hazard detection step.")
            return

        # Step 4: Localize in 3D
        if len(detections) > 0:
            hazard_locations = self.step4_localize_3d(detections)

            # Step 5: Visualize
            if len(hazard_locations) > 0:
                self.step5_visualize(hazard_locations)
        else:
            print("\nNo hazards detected. Skipping 3D localization.")

        print("\n" + "="*70)
        print("PIPELINE COMPLETE!")
        print("="*70)


def main():
    """
    Main entry point
    """
    pipeline = HazardLocPipeline()

    # Run with options
    pipeline.run_complete_pipeline(
        skip_data_prep=False,  # Set to True to skip downloading from Kaggle
        skip_preprocessing=False, # Set to True to skip splitting data into train/val
        skip_training=False,  # Set to True if model already trained
        skip_colmap=False     # Set to True if COLMAP already run
    )


if __name__ == '__main__':
    main()