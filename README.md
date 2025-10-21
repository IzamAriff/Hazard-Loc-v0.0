HazardLoc/
├── README.md # Project overview & setup instructions
├── environment.yml # Conda environment config (or requirements.txt for pip)
├── .gitignore # Git version control exclusions
├── data/
│ ├── raw/ # Original, unmodified images/datasets
│ ├── processed/ # Preprocessed/augmented data for modeling
│ └── cleaned/ # Final cleaned data ready for experiments
├── notebooks/
│ ├── 01-data_exploration.ipynb
│ ├── 02-model_training.ipynb
│ ├── 03-3d_reconstruction.ipynb
│ ├── 04-visualization.ipynb
├── src/
│ ├── init.py
│ ├── config.py # Central config/parameters
│ ├── main.py # End-to-end pipeline entrypoint
│ ├── train.py # Hazard detection model training & validation
│ ├── evaluate.py # Accuracy & performance metrics
│ ├── detect.py # Inference module for hazard localization
│ ├── reconstruct.py # COLMAP-based 3D reconstruction interface
│ ├── visualize.py # Open3D for point cloud/hazard visualization
│ ├── data/
│ │ ├── init.py
│ │ ├── dataloader.py # PyTorch Dataset/DataLoader for images
│ │ ├── preprocess.py # Preprocessing routines
│ │ └── augmentation.py # Augmentation logic
│ ├── models/
│ │ ├── init.py
│ │ ├── hazard_cnn.py # Custom & pretrained CNNs
│ │ └── pretrained.py # Fine-tuned/training initialization
│ ├── utils/
│ │ ├── init.py
│ │ ├── metrics.py
│ │ ├── logger.py
│ │ ├── colmap_utils.py # Adapter for COLMAP CLI/data
│ │ └── geo_map.py # Geolocation and mapping tools
│ ├── 3d/
│ │ ├── init.py
│ │ ├── colmap_adapter.py # Parsers/adapters for COLMAP outputs
│ │ ├── backproject.py # 2D to 3D backprojection logic
│ │ └── open3d_viz.py # Open3D visualization scripts
├── colmap/
│ ├── project.ini # COLMAP project settings/config
│ ├── database.db # COLMAP database file
│ ├── images/ # Input images for SfM workflow
│ └── output/
│ ├── cameras.txt
│ ├── images.txt
│ ├── points3D.txt
│ ├── stereo/
│ │ ├── consistency_graphs/
│ │ ├── depth_maps/
│ │ ├── normal_maps/
│ │ ├── patch-match.cfg
│ │ ├── fusion.cfg
│ ├── fused.ply
│ ├── meshed-poisson.ply
│ └── meshed-delaunay.ply
├── results/
│ ├── predictions.csv
│ ├── visualizations/
│ │ ├── demo_3d_pointcloud.png
│ │ └── hazard_map.png
├── tests/
│ ├── test_data.py
│ ├── test_model.py
│ ├── test_colmap.py
│ └── test_geo_map.py
├── docs/
│ ├── architecture.md
│ ├── pipeline.md
│ └── usage_guide.md
