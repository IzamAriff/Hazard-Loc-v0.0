# Central config/parameters

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR      = os.path.join(PROJECT_ROOT, "data")
MODEL_SAVE    = os.path.join(PROJECT_ROOT, "results", "hazard_cnn.pth")
COLMAP_IMG    = os.path.join(PROJECT_ROOT, "colmap", "images")
COLMAP_OUT    = os.path.join(PROJECT_ROOT, "colmap", "output")
VISUAL_DIR    = os.path.join(PROJECT_ROOT, "results", "visualizations")

# --- IMPORTANT ---
# UPDATE THIS PATH to point to your COLMAP.bat file after installation
# Point directly to the .exe. The utils will handle the path environment.
COLMAP_EXE = r"C:\Program Files\COLMAP-3.8-windows-cuda\bin\colmap.exe"
