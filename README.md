# HazardLoc: Sensor-less 3D Hazard Localization System

**HazardLoc** fuses deep learning (PyTorch CNN), Structure-from-Motion (COLMAP), and 3D visualization (Open3D) to localize hazards from 2D images—no extra sensors needed.

## Quickstart

- Place images in `colmap/images/`
  
- Train/evaluate:  
`python src/train.py`
`python src/evaluate.py`

- Run full pipeline:  
`python src/main.py`

