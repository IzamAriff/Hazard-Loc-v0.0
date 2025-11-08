# Open3D visualization scripts
"""
Open3D Visualization Module for HazardLoc
Visualizes 3D point clouds and hazard locations
"""

import open3d as o3d
import numpy as np
from pathlib import Path


def create_point_cloud_from_colmap(points3D):
    """
    Create Open3D point cloud from COLMAP points

    Args:
        points3D: Dictionary of 3D points from COLMAP

    Returns:
        Open3D PointCloud object
    """
    # Extract coordinates and colors
    xyz = np.array([p['xyz'] for p in points3D.values()])
    rgb = np.array([p['rgb'] for p in points3D.values()]) / 255.0

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    return pcd


def visualize_hazards_in_3d(point_cloud, hazard_locations, hazard_labels=None):
    """
    Visualize point cloud with hazard markers

    Args:
        point_cloud: Open3D PointCloud object
        hazard_locations: List of 3D coordinates of hazards
        hazard_labels: Optional labels for hazards
    """
    geometries = [point_cloud]

    # Create sphere markers for hazards
    for idx, loc in enumerate(hazard_locations):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        sphere.translate(loc)
        sphere.paint_uniform_color([1.0, 0.0, 0.0])  # Red for hazards
        geometries.append(sphere)

        # Add label if provided
        if hazard_labels and idx < len(hazard_labels):
            # Could add text labels here (requires additional implementation)
            pass

    # Add coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    geometries.append(coord_frame)

    # Visualize
    o3d.visualization.draw_geometries(
        geometries,
        window_name="HazardLoc 3D Visualization",
        width=1280,
        height=720,
        left=50,
        top=50
    )


def save_point_cloud(point_cloud, output_path):
    """
    Save point cloud to file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    o3d.io.write_point_cloud(str(output_path), point_cloud)
    print(f"Point cloud saved to: {output_path}")


def create_hazard_map(point_cloud, hazard_locations, output_path):
    """
    Create and save hazard visualization
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Hazard Map Capture", width=1920, height=1080, visible=False)

    vis.add_geometry(point_cloud)

    for loc in hazard_locations:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
        sphere.translate(loc)
        sphere.paint_uniform_color([1.0, 0.0, 0.0])
        vis.add_geometry(sphere)

    # Update the renderer to process the geometries, then adjust the camera view.
    # This sequence prevents the "SetViewPoint() failed" warning.
    view_control = vis.get_view_control()
    view_control.set_zoom(0.7)

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(str(output_path))
    vis.destroy_window()

    print(f"Hazard map saved to: {output_path}")


class HazardVisualizer:
    """
    Complete visualization pipeline for HazardLoc
    """

    def __init__(self, colmap_output_dir, colmap_image_dir):
        self.colmap_dir = Path(colmap_output_dir)
        self.colmap_image_dir = Path(colmap_image_dir)
        self.point_cloud = None
        self.hazard_locations = []

    def load_point_cloud(self):
        """Load point cloud from COLMAP output"""
        from src.utils.colmap_utils import read_colmap_outputs

        _, _, points3D = read_colmap_outputs(self.colmap_dir, self.colmap_image_dir)
        self.point_cloud = create_point_cloud_from_colmap(points3D)

        print(f"Loaded point cloud with {len(self.point_cloud.points)} points")
        return self.point_cloud

    def add_hazard(self, location):
        """Add hazard location to visualization"""
        self.hazard_locations.append(location)

    def visualize(self):
        """Display interactive visualization"""
        if self.point_cloud is None:
            raise ValueError("Point cloud not loaded. Call load_point_cloud() first.")

        visualize_hazards_in_3d(self.point_cloud, self.hazard_locations)

    def save_visualization(self, output_path):
        """Save visualization to image"""
        if self.point_cloud is None:
            raise ValueError("Point cloud not loaded.")

        create_hazard_map(self.point_cloud, self.hazard_locations, output_path)