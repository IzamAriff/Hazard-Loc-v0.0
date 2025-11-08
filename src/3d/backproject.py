"""
Backprojection and 3D Localization Module
"""

import numpy as np
from scipy.spatial.transform import Rotation

# --- Helper Functions ---

def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion (w, x, y, z) to a 3x3 rotation matrix.
    """
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

def get_projection_matrix(camera_data, image_data):
    """
    Constructs the 3x4 projection matrix P = K * [R|t] for an image.

    Args:
        camera_data (dict): Dictionary of all camera models from COLMAP.
        image_data (dict): Dictionary for a single image from COLMAP.

    Returns:
        P (np.ndarray): 3x4 projection matrix.
        K (np.ndarray): 3x3 camera intrinsic matrix.
        R (np.ndarray): 3x3 rotation matrix (camera-to-world).
        t (np.ndarray): 3x1 translation vector (camera-to-world).
    """
    camera_id = image_data['camera_id']
    if camera_id not in camera_data:
        raise ValueError(f"Camera ID {camera_id} for image {image_data['name']} not found in camera data.")

    camera = camera_data[camera_id]
    model = camera['model']
    params = camera['params']

    # 1. Intrinsics (K)
    if model == 'SIMPLE_PINHOLE':
        # f, cx, cy
        f, cx, cy = params
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    elif model == 'PINHOLE':
        # fx, fy, cx, cy
        fx, fy, cx, cy = params
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    elif model == 'SIMPLE_RADIAL':
        # f, cx, cy, k
        f, cx, cy, _ = params # Distortion 'k' is not needed for projection matrix
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    else:
        raise ValueError(f"Unsupported camera model: {camera['model']}")

    # 2. Extrinsics (R, t)
    # COLMAP provides camera-to-world transformation (C2W).
    # We need world-to-camera (W2C) for the projection matrix.
    # R_c2w, t_c2w
    q = image_data['quaternion']
    t_c2w = image_data['translation']
    R_c2w = quaternion_to_rotation_matrix(q)

    # Invert to get world-to-camera (W2C) extrinsics
    # R_w2c = R_c2w.T
    # t_w2c = -R_c2w.T @ t_c2w
    R_w2c = R_c2w.transpose()
    t_w2c = -R_w2c @ t_c2w

    # 3. Projection Matrix (P)
    # P = K @ [R_w2c | t_w2c]
    P = K @ np.hstack((R_w2c, t_w2c.reshape(3, 1)))

    return P, K, R_c2w, t_c2w


class HazardLocalizer:
    """
    Localizes 2D detections in a 3D sparse reconstruction.
    """

    def __init__(self, cameras, images, points3D):
        self.cameras = cameras
        self.points3D = points3D

        # Create a mapping from image name to image data for quick lookups
        self.image_name_to_data = {img['name']: img for img in images.values()}

        # Pre-calculate projection matrices for all images
        self.projection_matrices = {}
        for image_data in images.values():
            try:
                P, _, _, _ = get_projection_matrix(self.cameras, image_data)
                self.projection_matrices[image_data['name']] = P
            except ValueError as e:
                print(f"Warning: Could not create projection matrix for {image_data['name']}. Reason: {e}")

        # Create a set of all 3D point IDs for efficient lookup
        self.point_ids = set(self.points3D.keys())

    def _project_points_to_image(self, image_name):
        """
        Projects all 3D points into a specific image plane.

        Returns:
            A dictionary mapping 3D point IDs to their 2D (x, y) coordinates.
        """
        if image_name not in self.projection_matrices:
            return {}

        P = self.projection_matrices[image_name]
        projected_points = {}

        # Get all 3D points as a single NumPy array for vectorized operation
        point_ids = list(self.point_ids)
        points_3d_xyz = np.array([self.points3D[pid]['xyz'] for pid in point_ids])
        
        # Add homogeneous coordinate
        points_3d_homogeneous = np.hstack((points_3d_xyz, np.ones((len(points_3d_xyz), 1))))

        # Project all points at once
        points_2d_homogeneous = (P @ points_3d_homogeneous.T).T

        # Normalize to get pixel coordinates
        # Avoid division by zero for points at infinity
        valid_idx = np.where(points_2d_homogeneous[:, 2] != 0)
        points_2d = points_2d_homogeneous[valid_idx]
        points_2d[:, 0] /= points_2d[:, 2]
        points_2d[:, 1] /= points_2d[:, 2]
        
        # Create the mapping from point ID to 2D coordinate
        valid_point_ids = np.array(point_ids)[valid_idx]
        for i, point_id in enumerate(valid_point_ids):
            projected_points[point_id] = points_2d[i, :2]

        return projected_points

    def localize_hazard(self, image_name, bbox):
        """
        Finds the 3D location of a hazard given a 2D bounding box.

        Args:
            image_name (str): The filename of the image with the detection.
            bbox (list): The bounding box [x_min, y_min, x_max, y_max].

        Returns:
            The median 3D coordinate (np.ndarray) of the points within the bbox, or None.
        """
        if image_name not in self.image_name_to_data:
            print(f"Warning: Image {image_name} not found in reconstruction")
            return None

        image_data = self.image_name_to_data[image_name]
        
        # Project all 3D points onto this image
        projected_points = self._project_points_to_image(image_name)
        if not projected_points:
            return None

        # Find which 3D points fall inside the bounding box
        x_min, y_min, x_max, y_max = bbox
        points_in_bbox = []
        for point_id, (x, y) in projected_points.items():
            if x_min <= x <= x_max and y_min <= y <= y_max:
                points_in_bbox.append(self.points3D[point_id]['xyz'])

        if not points_in_bbox:
            return None

        # Return the median of the 3D points as the hazard location
        # Median is more robust to outliers than the mean.
        median_3d_point = np.median(np.array(points_in_bbox), axis=0)

        return median_3d_point
