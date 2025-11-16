"""
3D Reconstruction Quality Validation
Implements dissertation Phase 3: Reconstruction quality metrics (Section 3.5.2)
"""

import numpy as np
from typing import Dict, Tuple, List


class ReconstructionQualityValidator:
    """
    Validate SfM reconstruction quality
    Dissertation reference: Section 3.5.2, Figure 14
    """

    def __init__(self, 
                 reprojection_error_threshold: float = 1.0,
                 reconstruction_angle_threshold: float = 2.0,
                 track_length_threshold: int = 3):
        """
        Initialize quality validator

        Args:
            reprojection_error_threshold: Maximum acceptable pixel error (target: 1.0)
            reconstruction_angle_threshold: Minimum viewing angle in degrees (target: 2.0)
            track_length_threshold: Minimum number of images observing a point (target: 3)
        """
        self.reprojection_error_threshold = reprojection_error_threshold
        self.reconstruction_angle_threshold = reconstruction_angle_threshold
        self.track_length_threshold = track_length_threshold

    def compute_reprojection_error(self, 
                                  observed_points: np.ndarray,
                                  reprojected_points: np.ndarray) -> float:
        """
        Compute reprojection error: difference between observed and reprojected feature locations

        Dissertation: Section 3.5.2, Equation for Bundle Adjustment
        Target: < 1.0 pixel

        Args:
            observed_points: (N, 2) array of observed image points
            reprojected_points: (N, 2) array of reprojected points from 3D reconstruction

        Returns:
            Mean reprojection error in pixels
        """
        if len(observed_points) == 0:
            return 0.0

        errors = np.linalg.norm(observed_points - reprojected_points, axis=1)
        mean_error = np.mean(errors)

        return mean_error

    def compute_reconstruction_angles(self,
                                     camera_positions: List[np.ndarray],
                                     point_3d: np.ndarray) -> List[float]:
        """
        Compute triangulation angles for a 3D point

        Dissertation: Section 3.5.2
        Target angle: > 2 degrees for reliable depth

        Acute angles produce unstable depth estimates
        Wide angles provide stable estimates

        Args:
            camera_positions: List of camera center coordinates
            point_3d: (3,) 3D point coordinate

        Returns:
            List of viewing angles in degrees
        """
        angles = []

        for i in range(len(camera_positions)):
            for j in range(i + 1, len(camera_positions)):
                # Vectors from cameras to point
                v1 = point_3d - camera_positions[i]
                v2 = point_3d - camera_positions[j]

                # Angle between viewing rays
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle_rad = np.arccos(cos_angle)
                angle_deg = np.degrees(angle_rad)

                angles.append(angle_deg)

        return angles

    def assess_point_track_length(self,
                                 points_3d: Dict,
                                 min_track_length: int = 3) -> Dict:
        """
        Analyze point track lengths

        Dissertation: Section 3.5.2
        Points observed in many images receive more constraint during optimization

        Returns:
            Statistics about point track lengths
        """
        track_lengths = []

        for point_id, point_data in points_3d.items():
            # Assuming point_data contains 'track' or 'image_ids'
            track_length = len(point_data.get('track', []))
            if track_length == 0:
                track_length = len(point_data.get('image_ids', []))
            
            if track_length == 0:
                track_length = 1

            track_lengths.append(track_length)

        if not track_lengths:
            return {'error': 'No points'}

        return {
            'mean_track_length': np.mean(track_lengths),
            'min_track_length': np.min(track_lengths),
            'max_track_length': np.max(track_lengths),
            'median_track_length': np.median(track_lengths),
            'points_below_threshold': sum(1 for t in track_lengths if t < min_track_length),
            'total_points': len(track_lengths),
            'acceptable': np.mean(track_lengths) >= min_track_length
        }

    def validate_reconstruction(self,
                               reprojection_errors: np.ndarray,
                               reconstruction_angles: List[float],
                               track_lengths: Dict) -> Dict:
        """
        Comprehensive reconstruction quality assessment

        Dissertation: Section 3.5.2

        Returns:
            Overall quality assessment
        """
        # Reprojection error assessment
        mean_reproj_error = np.mean(reprojection_errors) if len(reprojection_errors) > 0 else 0
        reproj_ok = mean_reproj_error < self.reprojection_error_threshold
        reproj_score = max(0, 1.0 - (mean_reproj_error / self.reprojection_error_threshold))

        # Reconstruction angle assessment
        mean_angle = np.mean(reconstruction_angles) if reconstruction_angles else 0
        angle_ok = mean_angle > self.reconstruction_angle_threshold
        angle_score = min(1.0, mean_angle / self.reconstruction_angle_threshold)

        # Track length assessment
        track_ok = track_lengths.get('acceptable', False)
        track_score = (track_lengths.get('mean_track_length', 0) / 
                      max(1, self.track_length_threshold + 2))

        # Overall quality score (0-100)
        overall_score = (reproj_score * 0.4 + angle_score * 0.3 + track_score * 0.3) * 100

        # Overall acceptable if all metrics pass
        acceptable = reproj_ok and angle_ok and track_ok

        return {
            'reprojection_error': {
                'mean': float(mean_reproj_error),
                'threshold': self.reprojection_error_threshold,
                'acceptable': reproj_ok,
                'score': float(reproj_score)
            },
            'reconstruction_angles': {
                'mean': float(mean_angle),
                'threshold': self.reconstruction_angle_threshold,
                'acceptable': angle_ok,
                'score': float(angle_score)
            },
            'track_lengths': {
                **track_lengths,
                'score': float(track_score)
            },
            'overall_quality_score': float(overall_score),
            'acceptable': acceptable,
            'status': 'GOOD' if acceptable else 'POOR - Review reconstruction'
        }

    def print_report(self, validation: Dict):
        """Pretty print validation results"""
        print("\n" + "="*70)
        print("3D RECONSTRUCTION QUALITY ASSESSMENT")
        print("="*70)

        print(f"\nOverall Status: {validation['status']}")
        print(f"Quality Score: {validation['overall_quality_score']:.1f}/100")

        print(f"\n--- Reprojection Error (Dissertation 3.5.2) ---")
        re = validation['reprojection_error']
        print(f"  Mean Error: {re['mean']:.3f} px (threshold: {re['threshold']} px)")
        print(f"  Status: {'✓ PASS' if re['acceptable'] else '✗ FAIL'}")

        print(f"\n--- Reconstruction Angles (Dissertation 3.5.2) ---")
        ra = validation['reconstruction_angles']
        print(f"  Mean Angle: {ra['mean']:.1f}° (threshold: {ra['threshold']}°)")
        print(f"  Status: {'✓ PASS' if ra['acceptable'] else '✗ FAIL'}")

        print(f"\n--- Point Track Lengths (Dissertation 3.5.2) ---")
        tl = validation['track_lengths']
        print(f"  Mean: {tl['mean_track_length']:.1f} images")
        print(f"  Range: {tl['min_track_length']}-{tl['max_track_length']}")
        print(f"  Median: {tl['median_track_length']:.1f}")
        print(f"  Points below threshold: {tl['points_below_threshold']}")
        print(f"  Status: {'✓ PASS' if tl['acceptable'] else '✗ FAIL'}")

        print(f"\n" + "="*70)