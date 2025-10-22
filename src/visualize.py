# Open3D for point cloud/hazard visualization

import open3d as o3d
import os
from src.config import COLMAP_OUT

def show_point_cloud(file='points3D.ply'):
    ply_path = os.path.join(COLMAP_OUT, file)
    pcd = o3d.io.read_point_cloud(ply_path)
    o3d.visualization.draw_geometries([pcd])
if __name__ == '__main__':
    show_point_cloud()
