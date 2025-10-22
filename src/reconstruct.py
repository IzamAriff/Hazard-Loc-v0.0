# COLMAP-based 3D reconstruction interface

import subprocess
import os
from src.config import COLMAP_IMG, COLMAP_OUT

def run_colmap_reconstruction():
    os.makedirs(COLMAP_OUT, exist_ok=True)
    cmds = [
        f"colmap feature_extractor --database_path {COLMAP_OUT}/database.db --image_path {COLMAP_IMG}",
        f"colmap exhaustive_matcher --database_path {COLMAP_OUT}/database.db",
        f"colmap mapper --database_path {COLMAP_OUT}/database.db --image_path {COLMAP_IMG} --output_path {COLMAP_OUT}",
        f"colmap model_converter --input_path {COLMAP_OUT}/0 --output_path {COLMAP_OUT} --output_type TXT"
    ]
    for cmd in cmds:
        subprocess.run(cmd, shell=True)
    print("COLMAP 3D reconstruction completed.")
if __name__ == '__main__':
    run_colmap_reconstruction()
