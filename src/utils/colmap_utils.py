# Adapter for COLMAP CLI/data
"""
COLMAP Utilities for HazardLoc
Interfaces with COLMAP for Structure-from-Motion reconstruction
"""

import subprocess
import os
from pathlib import Path
import numpy as np


class COLMAPAdapter:
    """
    Wrapper for COLMAP command-line interface
    """

    def __init__(self, colmap_dir, images_dir):
        self.colmap_dir = Path(colmap_dir)
        self.images_dir = Path(images_dir)
        self.database_path = self.colmap_dir / "database.db"
        self.output_dir = self.colmap_dir / "output"
        self.sparse_dir = self.output_dir / "sparse" / "0"

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sparse_dir.mkdir(parents=True, exist_ok=True)

    def feature_extraction(self):
        """Extract features from images"""
        print("Step 1/4: Extracting features...")

        cmd = [
            "colmap", "feature_extractor",
            "--database_path", str(self.database_path),
            "--image_path", str(self.images_dir),
            "--ImageReader.single_camera", "1",
            "--SiftExtraction.use_gpu", "1" if self._has_gpu() else "0"
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✓ Feature extraction complete")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Feature extraction failed: {e.stderr}")
            return False

    def feature_matching(self):
        """Match features between images"""
        print("Step 2/4: Matching features...")

        cmd = [
            "colmap", "exhaustive_matcher",
            "--database_path", str(self.database_path),
            "--SiftMatching.use_gpu", "1" if self._has_gpu() else "0"
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✓ Feature matching complete")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Feature matching failed: {e.stderr}")
            return False

    def sparse_reconstruction(self):
        """Perform sparse 3D reconstruction"""
        print("Step 3/4: Sparse reconstruction...")

        cmd = [
            "colmap", "mapper",
            "--database_path", str(self.database_path),
            "--image_path", str(self.images_dir),
            "--output_path", str(self.output_dir / "sparse")
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✓ Sparse reconstruction complete")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Sparse reconstruction failed: {e.stderr}")
            return False

    def model_converter(self):
        """Convert binary model to text format"""
        print("Step 4/4: Converting model to text...")

        cmd = [
            "colmap", "model_converter",
            "--input_path", str(self.sparse_dir),
            "--output_path", str(self.output_dir),
            "--output_type", "TXT"
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✓ Model conversion complete")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Model conversion failed: {e.stderr}")
            return False

    def run_full_pipeline(self):
        """Execute complete COLMAP pipeline"""
        print("\n" + "="*60)
        print("COLMAP RECONSTRUCTION PIPELINE")
        print("="*60 + "\n")

        steps = [
            self.feature_extraction,
            self.feature_matching,
            self.sparse_reconstruction,
            self.model_converter
        ]

        for step in steps:
            if not step():
                print(f"\nPipeline failed at: {step.__name__}")
                return False

        print("\n" + "="*60)
        print("COLMAP pipeline completed successfully!")
        print(f"Output saved to: {self.output_dir}")
        print("="*60 + "\n")

        return True

    @staticmethod
    def _has_gpu():
        """Check if GPU is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

    def parse_cameras(self):
        """Parse cameras.txt file"""
        cameras_file = self.output_dir / "cameras.txt"
        cameras = {}

        if not cameras_file.exists():
            print(f"Warning: {cameras_file} not found")
            return cameras

        with open(cameras_file, 'r') as f:
            for line in f:
                if line.startswith('#') or line.strip() == '':
                    continue

                parts = line.strip().split()
                camera_id = int(parts[0])
                model = parts[1]
                width = int(parts[2])
                height = int(parts[3])
                params = [float(p) for p in parts[4:]]

                cameras[camera_id] = {
                    'model': model,
                    'width': width,
                    'height': height,
                    'params': params
                }

        return cameras

    def parse_images(self):
        """Parse images.txt file"""
        images_file = self.output_dir / "images.txt"
        images = {}

        if not images_file.exists():
            print(f"Warning: {images_file} not found")
            return images

        with open(images_file, 'r') as f:
            lines = [line for line in f if not line.startswith('#') and line.strip()]

            for i in range(0, len(lines), 2):
                parts = lines[i].strip().split()
                image_id = int(parts[0])
                qw, qx, qy, qz = [float(x) for x in parts[1:5]]
                tx, ty, tz = [float(x) for x in parts[5:8]]
                camera_id = int(parts[8])
                name = parts[9]

                images[image_id] = {
                    'name': name,
                    'camera_id': camera_id,
                    'quaternion': np.array([qw, qx, qy, qz]),
                    'translation': np.array([tx, ty, tz])
                }

        return images

    def parse_points3D(self):
        """Parse points3D.txt file"""
        points_file = self.output_dir / "points3D.txt"
        points = {}

        if not points_file.exists():
            print(f"Warning: {points_file} not found")
            return points

        with open(points_file, 'r') as f:
            for line in f:
                if line.startswith('#') or line.strip() == '':
                    continue

                parts = line.strip().split()
                point_id = int(parts[0])
                xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                rgb = np.array([int(parts[4]), int(parts[5]), int(parts[6])])
                error = float(parts[7])

                points[point_id] = {
                    'xyz': xyz,
                    'rgb': rgb,
                    'error': error
                }

        return points


def read_colmap_outputs(colmap_output_dir):
    """
    Convenience function to read all COLMAP outputs
    """
    adapter = COLMAPAdapter(colmap_output_dir, None)

    cameras = adapter.parse_cameras()
    images = adapter.parse_images()
    points3D = adapter.parse_points3D()

    print(f"Loaded {len(cameras)} cameras")
    print(f"Loaded {len(images)} images")
    print(f"Loaded {len(points3D)} 3D points")

    return cameras, images, points3D