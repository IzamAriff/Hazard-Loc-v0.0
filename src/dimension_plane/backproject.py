"""
Advanced Backprojection - CUDA ACCELERATED with MEMORY FIXES
Uses PyTorch + CUDA with intelligent batching for 4GB GPU
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from typing import Dict, List, Tuple, Optional


def quaternion_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix."""
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def get_projection_matrix(camera_data, image_data):
    """Constructs the 3x4 projection matrix."""
    camera_id = image_data['camera_id']
    if camera_id not in camera_data:
        raise ValueError(f"Camera ID {camera_id} not found")

    camera = camera_data[camera_id]
    model = camera['model']
    params = camera['params']

    if model == 'SIMPLE_PINHOLE':
        f, cx, cy = params
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
    elif model == 'PINHOLE':
        fx, fy, cx, cy = params
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    elif model == 'SIMPLE_RADIAL':
        f, cx, cy, _ = params
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported camera model: {model}")

    q = image_data['quaternion']
    t_c2w = image_data['translation']
    R_c2w = quaternion_to_rotation_matrix(q)

    R_w2c = R_c2w.transpose()
    t_w2c = -R_w2c @ t_c2w

    P = K @ np.hstack((R_w2c, t_w2c.reshape(3, 1)))

    return P, K, R_c2w, t_c2w


# ============================================================================
# CUDA ACCELERATED HAZARD LOCALIZER - MEMORY OPTIMIZED WITH BATCHING
# ============================================================================

class AdvancedHazardLocalizer:
    """
    Advanced 3D localization with CUDA acceleration and intelligent batching.
    Handles 4GB GPU by processing rays in small batches.
    """

    def __init__(self, cameras, images, points3D, mesh_path: Optional[str] = None, use_cuda: bool = True):
        """Initialize the CUDA-accelerated localizer with smart batching."""
        self.cameras = cameras
        self.points3D = points3D
        self.use_cuda = use_cuda and torch.cuda.is_available()

        # GPU MEMORY OPTIMIZATION: Batch size for ray processing
        # Adjust if you have more/less GPU memory
        self.batch_size_rays = 8  # Process 8 rays at a time

        if self.use_cuda:
            print(f"✓ CUDA enabled! Using: {torch.cuda.get_device_name(0)}")
            print(f"  Batch size: {self.batch_size_rays} rays per batch (memory-optimized)")
        else:
            print("⚠ CUDA disabled or not available. Using CPU.")

        # Create image lookup dictionary
        self.image_name_to_data = {img['name']: img for img in images.values()}

        # Pre-compute camera extrinsics
        self.camera_extrinsics = {}
        for image_name, image_data in self.image_name_to_data.items():
            try:
                _, K, R_c2w, t_c2w = get_projection_matrix(cameras, image_data)
                R_w2c = R_c2w.T
                t_w2c = -R_w2c @ t_c2w
                self.camera_extrinsics[image_name] = (R_w2c, t_w2c, K)
            except ValueError as e:
                print(f"Warning: Could not compute extrinsics for {image_name}: {e}")

        # Load mesh if provided
        self.mesh = None
        if mesh_path:
            try:
                import trimesh
                self.mesh = trimesh.load_mesh(mesh_path)
                print(f"✓ Loaded mesh from {mesh_path}")
            except Exception as e:
                print(f"Warning: Could not load mesh: {e}")

    # ========================================================================
    # SMART BATCHED CUDA INTERSECTION (MEMORY EFFICIENT)
    # ========================================================================

    def batch_ray_cloud_intersection_cuda_smart(self, rays: List[Tuple[np.ndarray, np.ndarray]], 
                                              points: np.ndarray, max_distance: float = 10.0) -> List[Optional[np.ndarray]]:
        """
        MEMORY-SMART: Process rays in small batches to fit in 4GB GPU
        Processes batch_size_rays at a time instead of all at once
        """
        if not self.use_cuda:
            return [self._ray_cloud_intersection_cpu(r[0], r[1], points, max_distance) for r in rays]

        num_rays = len(rays)
        all_results = []

        try:
            # Convert points to GPU once (they stay there)
            points_array = np.array(points, dtype=np.float32)
            points_gpu = torch.tensor(points_array, device='cuda')

            # Process rays in batches
            for batch_start in range(0, num_rays, self.batch_size_rays):
                batch_end = min(batch_start + self.batch_size_rays, num_rays)
                batch_rays = rays[batch_start:batch_end]

                try:
                    # Stack rays for this batch
                    ray_origins = np.array([r[0] for r in batch_rays], dtype=np.float32)
                    ray_directions = np.array([r[1] for r in batch_rays], dtype=np.float32)

                    # Move to GPU
                    ray_origins_gpu = torch.tensor(ray_origins, device='cuda')
                    ray_directions_gpu = torch.tensor(ray_directions, device='cuda')

                    # Compute for all rays in batch and all points
                    origins_expanded = ray_origins_gpu.unsqueeze(1)          # (batch_size, 1, 3)
                    points_expanded = points_gpu.unsqueeze(0)                # (1, num_points, 3)

                    to_points = points_expanded - origins_expanded           # (batch_size, num_points, 3)

                    directions_expanded = ray_directions_gpu.unsqueeze(1)    # (batch_size, 1, 3)
                    t_values = torch.sum(to_points * directions_expanded, dim=2)
                    t_values = torch.clamp(t_values, min=0)

                    closest_pts = (origins_expanded + 
                                  t_values.unsqueeze(2) * directions_expanded)

                    distances = torch.norm(
                        points_expanded - closest_pts, dim=2
                    )  # (batch_size, num_points)

                    # Find closest point for each ray in batch
                    for ray_idx in range(len(batch_rays)):
                        valid_mask = distances[ray_idx] <= max_distance
                        if torch.any(valid_mask):
                            closest_idx = torch.argmin(distances[ray_idx][valid_mask])
                            closest_point = points_gpu[valid_mask][closest_idx].cpu().numpy()
                            all_results.append(closest_point)
                        else:
                            all_results.append(None)

                    # Clear GPU memory after each batch
                    del ray_origins_gpu, ray_directions_gpu, origins_expanded, to_points, t_values, closest_pts, distances
                    torch.cuda.empty_cache()

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"⚠ Batch {batch_start//self.batch_size_rays + 1} OOM, reducing batch size to CPU fallback")
                        # Fallback to CPU for this batch
                        for ray in batch_rays:
                            result = self._ray_cloud_intersection_cpu(ray[0], ray[1], points, max_distance)
                            all_results.append(result)
                    else:
                        raise

            return all_results

        except Exception as e:
            print(f"Warning: CUDA batch error {e}, falling back to CPU")
            return [self._ray_cloud_intersection_cpu(r[0], r[1], points, max_distance) for r in rays]

    def _ray_cloud_intersection_cpu(self, ray_origin: np.ndarray, ray_direction: np.ndarray,
                                   points: np.ndarray, max_distance: float = 10.0) -> Optional[np.ndarray]:
        """Fallback CPU version (chunked, memory efficient)"""
        if len(points) == 0:
            return None

        closest_distance = float('inf')
        closest_point = None
        chunk_size = 5000
        num_points = len(points)

        for chunk_start in range(0, num_points, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_points)
            chunk = points[chunk_start:chunk_end]

            to_points = chunk - ray_origin
            t_values = np.dot(to_points, ray_direction)
            t_values = np.maximum(t_values, 0)

            for i, t in enumerate(t_values):
                closest_pt_on_ray = ray_origin + t * ray_direction
                distance = np.linalg.norm(chunk[i] - closest_pt_on_ray)

                if distance < closest_distance and distance < max_distance:
                    closest_distance = distance
                    closest_point = chunk[i]

        return closest_point if closest_point is not None else None

    # ========================================================================
    # STANDARD METHODS
    # ========================================================================

    def backproject_pixel_to_ray(self, image_name: str, pixel_u: float, pixel_v: float) -> Tuple[np.ndarray, np.ndarray]:
        """Back-project a pixel to 3D ray."""
        if image_name not in self.camera_extrinsics:
            raise ValueError(f"Image {image_name} not found")

        R_w2c, t_w2c, K = self.camera_extrinsics[image_name]
        R_c2w = R_w2c.T
        t_c2w = -R_c2w @ t_w2c

        K_inv = np.linalg.inv(K)
        pixel_homogeneous = np.array([pixel_u, pixel_v, 1.0], dtype=np.float32)
        ray_cam = K_inv @ pixel_homogeneous

        ray_direction_world = R_c2w @ ray_cam
        ray_direction_world = ray_direction_world / np.linalg.norm(ray_direction_world)

        return t_c2w, ray_direction_world

    def backproject_bbox_to_rays(self, image_name: str, bbox: List[float]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Back-project all pixels in bbox to rays."""
        x_min, y_min, x_max, y_max = bbox
        rays = []

        for y in range(int(np.ceil(y_min)), int(np.floor(y_max)) + 1):
            for x in range(int(np.ceil(x_min)), int(np.floor(x_max)) + 1):
                try:
                    ray_origin, ray_direction = self.backproject_pixel_to_ray(image_name, x, y)
                    rays.append((ray_origin, ray_direction))
                except:
                    pass

        return rays

    def localize_hazard_advanced(self, image_name: str, bbox: List[float], 
                               detection_confidence: float = 0.8) -> Optional[Dict]:
        """Advanced hazard localization with memory-smart CUDA acceleration."""
        if image_name not in self.camera_extrinsics:
            return None

        # Get rays
        rays = self.backproject_bbox_to_rays(image_name, bbox)
        if len(rays) == 0:
            return None

        # Get points array
        points = np.array([self.points3D[pid]['xyz'] for pid in self.points3D.keys()], dtype=np.float32)

        # Use SMART BATCHED CUDA for rays (fits in 4GB GPU!)
        crack_locations_3d = self.batch_ray_cloud_intersection_cuda_smart(rays, points, max_distance=10.0)
        crack_locations_3d = [p for p in crack_locations_3d if p is not None]

        if len(crack_locations_3d) == 0:
            return None

        crack_locations_array = np.array(crack_locations_3d, dtype=np.float32)
        median_location = np.median(crack_locations_array, axis=0)

        return {
            'location_3d': median_location,
            'image_name': image_name,
            'bbox': bbox,
            'confidence': detection_confidence,
            'crack_locations_3d': crack_locations_3d,
            'num_rays': len(rays),
            'num_intersections': len(crack_locations_3d),
        }

    def cluster_points_3d(self, points_3d: List[np.ndarray], 
                         clustering_radius: float = 0.05) -> List[List[int]]:
        """Cluster 3D points based on spatial proximity."""
        if len(points_3d) == 0:
            return []

        points_array = np.array(points_3d, dtype=np.float32)
        tree = cKDTree(points_array)
        clusters = []
        visited = set()

        for i in range(len(points_3d)):
            if i in visited:
                continue

            neighbor_indices = tree.query_ball_point(points_3d[i], clustering_radius)
            for idx in neighbor_indices:
                visited.add(idx)
            clusters.append(neighbor_indices)

        return clusters

    def compute_weighted_centroid(self, points_3d: List[np.ndarray], 
                                 detection_confidences: List[float],
                                 viewing_angles: List[float]) -> np.ndarray:
        """Compute weighted average of 3D points within a cluster."""
        if len(points_3d) == 0:
            return None

        points_array = np.array(points_3d, dtype=np.float32)
        viewing_angle_weights = np.array([(np.cos(angle) + 1) / 2 for angle in viewing_angles], dtype=np.float32)
        weights = np.array(detection_confidences, dtype=np.float32) * viewing_angle_weights
        weights = weights / np.sum(weights)

        weighted_centroid = np.sum(points_array * weights[:, np.newaxis], axis=0)
        return weighted_centroid

    def fuse_multiview_detections(self, detections_per_image: Dict[str, Dict]) -> List[Dict]:
        """Fuse multi-view detections through clustering and weighting."""
        all_3d_points = []
        point_metadata = []

        for image_name, detection in detections_per_image.items():
            confidence = detection.get('confidence', 0.5)

            for point_3d in detection.get('crack_locations_3d', []):
                all_3d_points.append(point_3d)

                if image_name in self.camera_extrinsics:
                    _, t_w2c, _ = self.camera_extrinsics[image_name]
                    viewing_angle = 0.1
                else:
                    viewing_angle = 0.1

                point_metadata.append({
                    'confidence': confidence,
                    'viewing_angle': viewing_angle,
                    'image_name': image_name
                })

        if len(all_3d_points) == 0:
            return []

        clusters = self.cluster_points_3d(all_3d_points, clustering_radius=0.05)
        fused_locations = []

        for cluster_indices in clusters:
            if len(cluster_indices) == 0:
                continue

            cluster_points = [all_3d_points[i] for i in cluster_indices]
            cluster_confidences = [point_metadata[i]['confidence'] for i in cluster_indices]
            cluster_viewing_angles = [point_metadata[i]['viewing_angle'] for i in cluster_indices]

            centroid = self.compute_weighted_centroid(cluster_points, cluster_confidences, cluster_viewing_angles)

            fused_locations.append({
                'location_3d': centroid,
                'cluster_size': len(cluster_indices),
                'avg_confidence': np.mean(cluster_confidences),
                'num_views': len(set(point_metadata[i]['image_name'] for i in cluster_indices)),
                'cluster_indices': cluster_indices
            })

        return fused_locations