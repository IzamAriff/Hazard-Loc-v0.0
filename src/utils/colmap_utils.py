# Adapter for COLMAP CLI/data
"""
COLMAP Utilities for HazardLoc
Interfaces with COLMAP for Structure-from-Motion reconstruction
"""

import subprocess
import os
from pathlib import Path
import sys
import numpy as np

from src.config import COLMAP_EXE


class COLMAPAdapter:
    """
    Wrapper for COLMAP command-line interface
    """

    def __init__(self, colmap_dir, images_dir):
        self.colmap_dir = Path(colmap_dir)
        self.images_dir = Path(images_dir) if images_dir else None
        self.database_path = self.colmap_dir / "database.db"
        # The colmap_dir is the output directory
        self.sparse_dir = self.colmap_dir / "sparse" / "0"

        # Create directories
        self.colmap_dir.mkdir(parents=True, exist_ok=True)
        self.sparse_dir.mkdir(parents=True, exist_ok=True)

    def _run_colmap_command(self, args):
        """
        Executes a COLMAP command with a modified environment to handle paths correctly on Windows.
        This is the definitive fix for issues with spaces in 'Program Files'.
        """
        # Build the full command
        cmd = [COLMAP_EXE] + args

        # Create a copy of the current environment
        env = os.environ.copy()

        # If on Windows, prepend COLMAP's bin and lib dirs to the PATH.
        # This replicates the behavior of COLMAP.bat and ensures DLLs are found.
        if sys.platform == 'win32':
            colmap_bin_dir = str(Path(COLMAP_EXE).parent)
            colmap_lib_dir = str(Path(colmap_bin_dir).parent / 'lib')
            env['PATH'] = f"{colmap_bin_dir};{colmap_lib_dir};{env['PATH']}"

        try:
            # Run the command with the modified environment, no shell=True needed.
            subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8', env=env)
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Command failed. COLMAP stderr:\n---_BEGIN_---\n{e.stderr.strip()}\n---_END_---")
            if e.stdout:
                print(f"COLMAP stdout:\n---_BEGIN_---\n{e.stdout.strip()}\n---_END_---")
            return False

    def feature_extraction(self):
        """Extract features from images"""
        print("Step 1/4: Extracting features...")
        args = [
            'feature_extractor',
            '--database_path', str(self.database_path),
            '--image_path', str(self.images_dir),
            '--ImageReader.single_camera', '1',
            '--SiftExtraction.use_gpu', "1" if self._has_gpu() else "0"
        ]
        if self._run_colmap_command(args):
            print("✓ Feature extraction complete")
            return True
        else:
            print("✗ Feature extraction failed.")
            return False

    def feature_matching(self):
        """Match features between images"""
        print("Step 2/4: Matching features...")
        args = [
            'exhaustive_matcher',
            '--database_path', str(self.database_path),
            '--SiftMatching.use_gpu', "1" if self._has_gpu() else "0"
        ]
        if self._run_colmap_command(args):
            print("✓ Feature matching complete")
            return True
        else:
            print("✗ Feature matching failed.")
            return False

    def sparse_reconstruction(self):
        """Perform sparse 3D reconstruction"""
        print("Step 3/4: Sparse reconstruction...")
        args = [
            'mapper',
            '--database_path', str(self.database_path),
            '--image_path', str(self.images_dir),
            '--output_path', str(self.colmap_dir / "sparse")
        ]
        if self._run_colmap_command(args):
            # --- VERIFICATION STEP ---
            # Check if the sparse model files were actually created.
            # The mapper can exit successfully without creating a model if reconstruction fails.
            expected_files = ["cameras.bin", "images.bin", "points3D.bin"]
            if all((self.sparse_dir / f).exists() for f in expected_files):
                print("✓ Sparse reconstruction complete")
                return True
            else:
                print("✗ Sparse reconstruction failed: Mapper did not generate model files.")
                print("  This often happens if there are not enough matching features between images.")
                print("  Try using more images or images with better overlap.")
                return False
        else:
            print("✗ Sparse reconstruction failed.")
            return False

    def model_converter(self):
        """Convert binary model to text format"""
        print("Step 4/4: Converting model to text...")
        args = [
            'model_converter',
            '--input_path', str(self.sparse_dir),
            '--output_path', str(self.sparse_dir),
            '--output_type', 'TXT'
        ]
        if self._run_colmap_command(args):
            # --- VERIFICATION STEP ---
            # Check if the text model files were actually created.
            expected_files = ["cameras.txt", "images.txt", "points3D.txt"]
            if all((self.sparse_dir / f).exists() for f in expected_files):
                print("✓ Model conversion complete")
                return True
            else:
                print("✗ Model conversion failed: Text files were not generated.")
                print("  This can happen if the sparse reconstruction produced no model.")
                return False
        else:
            print("✗ Model conversion failed.")
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
        print(f"Output saved to: {self.colmap_dir}")
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
        cameras_file = self.sparse_dir / "cameras.txt"
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
        images_file = self.sparse_dir / "images.txt"
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
        points_file = self.sparse_dir / "points3D.txt"
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


def read_colmap_outputs(colmap_base_dir, colmap_image_dir):
    """
    Convenience function to read all COLMAP outputs
    """
    adapter = COLMAPAdapter(colmap_base_dir, colmap_image_dir)

    cameras = adapter.parse_cameras()
    images = adapter.parse_images()
    points3D = adapter.parse_points3D()

    print(f"Loaded {len(cameras)} cameras")
    print(f"Loaded {len(images)} images")
    print(f"Loaded {len(points3D)} 3D points")

    return cameras, images, points3D