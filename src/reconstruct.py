'''
## COLMAP-based 3D reconstruction interface

import subprocess
import os
from src.config import COLMAP_IMG, COLMAP_OUT, COLMAP_EXE

def run_colmap_reconstruction():
    os.makedirs(COLMAP_OUT, exist_ok=True)

    # Ensure the COLMAP executable path is valid
    if not os.path.exists(COLMAP_EXE):
        raise FileNotFoundError(f"COLMAP executable not found at: {COLMAP_EXE}. Please update the path in src/config.py")

    cmds = [
        f'"{COLMAP_EXE}" feature_extractor --database_path "{COLMAP_OUT}/database.db" --image_path "{COLMAP_IMG}"',
        f'"{COLMAP_EXE}" exhaustive_matcher --database_path "{COLMAP_OUT}/database.db"',
        f'"{COLMAP_EXE}" mapper --database_path "{COLMAP_OUT}/database.db" --image_path "{COLMAP_IMG}" --output_path "{COLMAP_OUT}"',
        f'"{COLMAP_EXE}" model_converter --input_path "{COLMAP_OUT}/0" --output_path "{COLMAP_OUT}/points" --output_type TXT'
    ]
    for cmd in cmds:
        subprocess.run(cmd, shell=True)
    print("COLMAP 3D reconstruction completed.")
if __name__ == '__main__':
    run_colmap_reconstruction()

'''
