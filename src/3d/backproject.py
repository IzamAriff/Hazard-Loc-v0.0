# 2D to 3D backprojection logic

"""
2D to 3D Backprojection Module for HazardLoc
Projects 2D hazard detections into 3D space using camera parameters
"""

import numpy as np
from scipy.spatial.transform import Rotation


def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion to rotation matrix

    Args:
        q: Quaternion as [qw, qx, qy, qz]

    Returns:
        3x3 rotation matrix
    """
    r = Rotation.from_quat([q[1], q[2], q[3], q[0]])  # scipy uses [qx,qy,qz,qw]
    return r.as_matrix()


def get_projection_matrix(camera, image):
    """
    Compute projection matrix from camera and image parameters

    Args:
        camera: Camera parameters dict
        image: Image parameters dict (pose)

    Returns:
        3x4 projection matrix
    """
    # Camera intrinsics
    if camera['model'] == 'SIMPLE_PINHOLE':
        f = camera['params'][0]
        cx = camera['params'][1]
        cy = camera['params'][2]
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ])
    elif camera['model'] == 'PINHOLE':
        fx = camera['params'][0]
        fy = camera['params'][1]
        cx = camera['params'][2]
        cy = camera['params'][3]
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
    else:
        raise ValueError(f"Unsupported camera model: {camera['model']}")

    # Camera extrinsics
    R = quaternion_to_rotation_matrix(image['quaternion'])
    t = image['translation'].reshape(3, 1)

    # Extrinsic matrix [R|t]
    extrinsic = np.hstack([R, t])

    # Projection matrix P = K[R|t]
    P = K @ extrinsic

    return P, K, R, t


def backproject_2d_to_3d(pixel_coords, depth, K, R, t):
    """
    Backproject 2D pixel coordinates to 3D world coordinates

    Args:
        pixel_coords: (u, v) pixel coordinates
        depth: Estimated depth at that pixel
        K: Camera intrinsic matrix
        R: Rotation matrix
        t: Translation vector

    Returns:
        3D world coordinates (x, y, z)
    """
    u, v = pixel_coords

    # Convert to homogeneous coordinates
    pixel_hom = np.array([u, v, 1.0])

    # Backproject to camera coordinates
    camera_coords = depth * np.linalg.inv(K) @ pixel_hom

    # Transform to world coordinates
    # X_world = R^T * (X_camera - t)
    world_coords = R.T @ (camera_coords - t.flatten())

    return world_coords


def triangulate_point(pixel_coords_list, camera_list, image_list):
    """
    Triangulate 3D point from multiple 2D observations

    Args:
        pixel_coords_list: List of (u,v) pixel coordinates
        camera_list: List of camera parameters
        image_list: List of image pose parameters

    Returns:
        Triangulated 3D point
    """
    if len(pixel_coords_list) < 2:
        raise ValueError("Need at least 2 views for triangulation")

    # Build linear system AX = 0
    A = []
    for pixel, camera, image in zip(pixel_coords_list, camera_list, image_list):
        P, _, _, _ = get_projection_matrix(camera, image)
        u, v = pixel

        # Each observation contributes 2 equations
        A.append(u * P[2, :] - P[0, :])
        A.append(v * P[2, :] - P[1, :])

    A = np.array(A)

    # Solve using SVD
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]

    # Convert from homogeneous coordinates
    X = X[:3] / X[3]

    return X


class HazardLocalizer:
    """
    Localize hazards in 3D space from 2D detections
    """

    def __init__(self, cameras, images, points3D):
        self.cameras = cameras
        self.images = images
        self.points3D = points3D

    def localize_hazard(self, image_name, bbox, depth_estimate=None):
        """
        Localize hazard from 2D bounding box

        Args:
            image_name: Name of the image
            bbox: Bounding box [x1, y1, x2, y2]
            depth_estimate: Optional depth estimate

        Returns:
            3D coordinates of hazard center
        """
        # Find image in reconstruction
        image_data = None
        for img_id, img in self.images.items():
            if img['name'] == image_name:
                image_data = img
                break

        if image_data is None:
            print(f"Warning: Image {image_name} not found in reconstruction")
            return None

        camera_data = self.cameras.get(image_data['camera_id'])
        if camera_data is None:
            print(f"Warning: Camera {image_data['camera_id']} not found")
            return None

        # Compute center of bounding box
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2

        # Get camera parameters
        P, K, R, t = get_projection_matrix(camera_data, image_data)

        # Estimate depth if not provided
        if depth_estimate is None:
            # Use median depth from nearby 3D points
            depth_estimate = self._estimate_depth_from_points(cx, cy, P)

        # Backproject to 3D
        world_coords = backproject_2d_to_3d((cx, cy), depth_estimate, K, R, t)

        return world_coords

    def _estimate_depth_from_points(self, u, v, P):
        """
        Estimate depth from nearby 3D points
        """
        depths = []

        for point_id, point in self.points3D.items():
            # Project 3D point to 2D
            X = np.append(point['xyz'], 1)
            x = P @ X
            x = x[:2] / x[2]

            # Check if close to our pixel
            dist = np.sqrt((x[0] - u)**2 + (x[1] - v)**2)
            if dist < 50:  # Within 50 pixels
                depth = np.linalg.norm(point['xyz'])
                depths.append(depth)

        if len(depths) > 0:
            return np.median(depths)
        else:
            return 5.0  # Default depth estimate