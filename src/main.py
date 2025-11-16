"""
HazardLoc: Complete Integration Pipeline
Runs end-to-end workflow from 2D detection to 3D localization
UPDATED: Uses AdvancedHazardLocalizer with multi-view fusion
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

# Import all modules
from src.data.dataloader import get_dataloaders
from src.data.downloader import download_from_kaggle
from src.data.preprocess import create_processed_dataset
from src.utils.losses import CompoundLoss
from src.models.hazard_cnn import HazardCNN
from src.train import HazardTrainer
from src.utils.colmap_utils import COLMAPAdapter, read_colmap_outputs

# NEW: Import advanced localizer (replaces old backproject)
from src.dimension_plane.backproject import AdvancedHazardLocalizer

# Import visualization
import importlib
open3d_viz_module = importlib.import_module("src.dimension_plane.open3d_viz")
HazardVisualizer = open3d_viz_module.HazardVisualizer

from src.detect import predict_img
from src.config import DATA_DIR, MODEL_SAVE, COLMAP_IMG, COLMAP_OUT, VISUAL_DIR, KAGGLE_DATASET_SLUG


class HazardLocPipeline:
    """
    End-to-end pipeline for HazardLoc system
    Now with advanced 3D localization using ray casting and multi-view fusion
    """

    def __init__(self):
        # --- Device detection ---
        try:
            import torch_xla.core.xla_model as xm
            self.device = xm.xla_device()
            print(f"✓ TPU device found: {xm.xla_real_devices([str(self.device)])[0]}")
        except ImportError:
            print("torch_xla not found, defaulting to CUDA/CPU.")
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Using device: {self.device}")

        self.model = None
        self.colmap = None
        self.localizer = None
        self.visualizer = None

        # NEW: Store detections for multi-view fusion
        self.all_detections = {}

    def step0_prepare_data(self, force_download=False):
        """
        Step 0: Download and prepare data from Kaggle
        """
        print("\n" + "="*70)
        print("STEP 0: PREPARING DATASET FROM KAGGLE")
        print("="*70)

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
                'batch_size': 32,
                'learning_rate': 1e-3,
                'weight_decay': 1e-4,
                'patience': 15,
                'use_bf16': True,
                'num_workers': 2,
                'loss_alpha': 1.0,
                'loss_beta': 0.5
            }

        # Automatic Learning Rate Scaling for TPU
        learning_rate = config['learning_rate']
        is_tpu = 'xla' in str(self.device)
        if is_tpu:
            num_cores = 8
            try:
                import torch_xla.core.xla_model as xm
                num_cores = xm.xrt_world_size()
            except (ImportError, AttributeError):
                try:
                    from torch_xla.distributed import xla_env
                    num_cores = xla_env.xrt_world_size()
                except (ImportError, AttributeError):
                    print("⚠ WARNING: Could not dynamically determine TPU core count.")
            learning_rate *= num_cores
            print(f"✓ TPU detected. Scaling learning rate by {num_cores} cores to: {learning_rate:.1e}")

        # Load data
        data = get_dataloaders(f"{DATA_DIR}/processed", batch_size=config['batch_size'], num_workers=config['num_workers'])

        # Create and train model
        num_classes = len(data['class_names'])
        self.model = HazardCNN(num_classes=num_classes)

        criterion = CompoundLoss(
            alpha_weight=config['loss_alpha'],
            beta_weight=config['loss_beta']
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=config['weight_decay']
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
        Returns detections organized by image name (for multi-view fusion)
        """
        print("\n" + "="*70)
        print("STEP 3: DETECTING HAZARDS IN IMAGES")
        print("="*70)

        detections = {}  # NEW: Dict by image name instead of list

        try:
            for img_path in test_images:
                label, confidence = predict_img(img_path)

                if label == 1:  # Hazard detected
                    from PIL import Image
                    with Image.open(img_path) as img:
                        width, height = img.size

                    # NEW: Store with image name as key
                    img_name = img_path.name
                    detections[img_name] = {
                        'image_path': str(img_path),
                        'confidence': confidence,
                        'bbox': [0, 0, width, height]
                    }
                    print(f"  ✓ Hazard detected in {img_name} (confidence: {confidence:.2%})")
                else:
                    print(f"  - No hazard in {img_path.name}")
        except FileNotFoundError as e:
            print(f"\n✗ ERROR: {e}")
            print("  Cannot run hazard detection. Halting pipeline.")
            return None

        print(f"\nTotal hazards detected: {len(detections)}")

        # Store for multi-view fusion
        self.all_detections = detections

        return detections

    def step4_localize_3d(self, detections: Dict) -> List[Dict]:
        """
        Step 4: Localize hazards in 3D space using AdvancedHazardLocalizer

        Implements dissertation Section 3.6.1:
          Step 1: Ray casting for each crack pixel
          Step 2: Ray-surface intersection with mesh
          Step 3: Multi-view fusion with clustering and weighting

        Returns: List of fused 3D hazard locations
        """
        print("\n" + "="*70)
        print("STEP 4: LOCALIZING HAZARDS IN 3D (ADVANCED)")
        print("="*70)

        try:
            # Load COLMAP outputs
            cameras, images, points3D = read_colmap_outputs(COLMAP_OUT, COLMAP_IMG)

            # Generate mesh path
            mesh_path = Path(COLMAP_OUT) / "dense" / "mesh.ply"
            if not mesh_path.exists():
                print(f"⚠ WARNING: Mesh not found at {mesh_path}")
                print("  Falling back to point cloud-based localization")
                mesh_path = None

            # Initialize ADVANCED localizer
            print("\nInitializing AdvancedHazardLocalizer...")
            self.localizer = AdvancedHazardLocalizer(
                cameras=cameras,
                images=images,
                points3D=points3D,
                mesh_path=str(mesh_path) if mesh_path else None,
                use_cuda=True
            )
            print("✓ AdvancedHazardLocalizer initialized")
        except Exception as e:
            print(f"\n✗ ERROR: Failed to initialize 3D localizer: {e}")
            return [] # Return empty list to indicate failure

        # STEP 4A: Single-image localization for each detection
        print("\n--- STEP 4A: Ray Casting & Surface Intersection ---")
        single_view_results = {}

        for image_name, detection in detections.items():
            try:
                print(f"\n  Processing: {image_name}")
                result = self.localizer.localize_hazard_advanced(
                    image_name=image_name,
                    bbox=detection['bbox'],
                    detection_confidence=detection['confidence']
                )

                if result is not None:
                    single_view_results[image_name] = result
                    print(f"    ✓ 3D location: {result['location_3d']}")
                    print(f"    ✓ Intersections: {result['num_intersections']}")
                else:
                    print(f"    ✗ Failed to localize")
            except Exception as e:
                print(f"    ✗ Error: {e}")

        if len(single_view_results) == 0:
            print("\n✗ No hazards successfully localized")
            return []

        # STEP 4B: Multi-view fusion
        print("\n--- STEP 4B: Multi-View Fusion ---")

        try:
            print(f"Fusing detections from {len(single_view_results)} views...")
            fused_locations = self.localizer.fuse_multiview_detections(single_view_results)

            print(f"\n✓ Multi-view fusion complete")
            print(f"  Fused locations: {len(fused_locations)}")

            for i, location in enumerate(fused_locations):
                print(f"\n  Fused Crack #{i+1}:")
                print(f"    3D Location: {location['location_3d']}")
                print(f"    Cluster Size: {location['cluster_size']}")
                print(f"    Avg Confidence: {location['avg_confidence']:.2%}")
                print(f"    Views: {location['num_views']}")

            return fused_locations
        except Exception as e:
            print(f"\n✗ Error during multi-view fusion: {e}")
            # Fallback to single-view results
            print("  Falling back to single-view results")
            return [{'location_3d': result['location_3d'], 'confidence': result['confidence']} 
                    for result in single_view_results.values()]

    def step5_visualize(self, fused_locations: List[Dict]):
        """
        Step 5: Visualize hazards in 3D point cloud
        """
        print("\n" + "="*70)
        print("STEP 5: VISUALIZING 3D HAZARD MAP")
        print("="*70)

        self.visualizer = HazardVisualizer(COLMAP_OUT, COLMAP_IMG)
        self.visualizer.load_point_cloud()

        # Add all fused hazard locations
        for loc in fused_locations:
            self.visualizer.add_hazard(loc['location_3d'])

        # Save visualization
        output_path = Path(VISUAL_DIR) / "hazard_map_3d.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.visualizer.save_visualization(output_path)

        print(f"\n✓ Visualization saved to: {output_path}")

        # Interactive view
        print("\nLaunching interactive 3D viewer...")
        self.visualizer.visualize()

    def step6_export_results(self, fused_locations: List[Dict]):
        """
        Step 6: Export results to CSV and JSON
        """
        print("\n" + "="*70)
        print("STEP 6: EXPORTING RESULTS")
        print("="*70)

        import json
        import csv

        results_dir = Path(VISUAL_DIR) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Export to CSV
        csv_path = results_dir / "hazard_locations_3d.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Crack_ID', 'X', 'Y', 'Z', 'Cluster_Size', 'Avg_Confidence', 'Num_Views'])
            for i, loc in enumerate(fused_locations):
                x, y, z = loc['location_3d']
                writer.writerow([
                    i + 1,
                    f"{x:.4f}",
                    f"{y:.4f}",
                    f"{z:.4f}",
                    loc.get('cluster_size', 'N/A'),
                    f"{loc.get('avg_confidence', 0):.2%}",
                    loc.get('num_views', 'N/A')
                ])
        print(f"✓ Results saved to CSV: {csv_path}")

        # Export to JSON
        json_path = results_dir / "hazard_locations_3d.json"
        json_data = {
            'total_cracks': len(fused_locations),
            'detections': []
        }
        for i, loc in enumerate(fused_locations):
            json_data['detections'].append({
                'crack_id': i + 1,
                'location_3d': loc['location_3d'].tolist() if isinstance(loc['location_3d'], np.ndarray) else loc['location_3d'],
                'cluster_size': int(loc.get('cluster_size', 0)),
                'avg_confidence': float(loc.get('avg_confidence', 0)),
                'num_views': int(loc.get('num_views', 0))
            })

        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"✓ Results saved to JSON: {json_path}")

    def run_complete_pipeline(self, skip_data_prep=False, skip_preprocessing=False, 
                            skip_training=False, skip_colmap=False):
        """
        Execute complete HazardLoc pipeline with advanced 3D localization
        """
        print("\n" + "="*70)
        print("HAZARDLOC: COMPLETE PIPELINE EXECUTION")
        print("Advanced 3D Localization with Multi-View Fusion")
        print("="*70)

        # Step 0: Prepare data from Kaggle
        if not skip_data_prep:
            if not self.step0_prepare_data():
                return
        else:
            print("\nSkipping data preparation (assuming data already exists)")

        # Step 0.5: Preprocess data
        if not skip_preprocessing:
            if not self.step0_5_preprocess_data():
                return
        else:
            print("\nSkipping data preprocessing (assuming data is already processed)")

        model_path = Path(MODEL_SAVE)

        # Step 1: Train model
        if skip_training:
            if not model_path.exists():
                print(f"⚠ WARNING: 'skip_training' is True, but model not found at '{model_path}'.")
                print("           Training is required. Forcing training step...")
                self.step1_train_detector()
            else:
                print(f"\nSkipping training (using existing model at '{model_path}').")
        else:
            print("\nStarting model training...")
            self.step1_train_detector()
 
        # Step 2: COLMAP reconstruction
        if not skip_colmap:
            success = self.step2_run_colmap()
            if not success:
                print("\nHalting pipeline due to COLMAP failure.")
                return
        else:
            print("\nSkipping COLMAP (using existing reconstruction).")

        # Step 3: Detect hazards
        all_test_images = sorted(list(Path(COLMAP_IMG).glob("*.jpg")))
        test_images = all_test_images[:10] # Using only the first 10 images
        print(f"\nRunning detection on the first {len(test_images)} of {len(all_test_images)} total images.")

        detections = self.step3_detect_hazards(test_images)
        if detections is None:
            print("\nHalting pipeline due to error in hazard detection step.")
            return

        # Step 4: Localize in 3D (ADVANCED with multi-view fusion)
        if len(detections) > 0:
            fused_locations = self.step4_localize_3d(detections)

            # Step 5: Visualize
            if len(fused_locations) > 0:
                self.step5_visualize(fused_locations)

                # Step 6: Export results
                self.step6_export_results(fused_locations)
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
        skip_data_prep=False,      # Set to False to download from Kaggle
        skip_preprocessing=False,  # Set to False to preprocess data
        skip_training=True,       # Set to False to train model
        skip_colmap=False         # Set to False to run COLMAP
    )

if __name__ == '__main__':
    main()
 