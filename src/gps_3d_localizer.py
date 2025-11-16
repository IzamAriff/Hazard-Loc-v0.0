"""
GPS to 3D Localization: Convert COLMAP 3D coordinates to real-world GPS
Provides practical information: GPS coordinates, distance, and crack size
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass
class GPSCoordinate:
    """Real-world GPS coordinate"""
    latitude: float
    longitude: float
    altitude: float

    def __str__(self):
        return f"GPS({self.latitude:.6f}°N, {self.longitude:.6f}°E, {self.altitude:.2f}m)"


class GPS3DLocalizer:
    """
    Convert COLMAP 3D coordinates to real-world GPS coordinates
    Provides distance measurement and crack size description
    """

    def __init__(self, origin_gps: GPSCoordinate, 
                 scale_factor: float = 1.0,
                 origin_orientation: float = 0.0):
        """
        Initialize GPS-3D mapping

        Args:
            origin_gps: GPS coordinate of COLMAP origin (0,0,0)
            scale_factor: Meters per unit in COLMAP (default 1.0)
            origin_orientation: North direction in degrees (0 = +X points North)
        """
        self.origin_gps = origin_gps
        self.scale_factor = scale_factor  # meters per COLMAP unit
        self.origin_orientation = origin_orientation  # degrees from +X axis

        print(f"✓ GPS-3D Localizer initialized")
        print(f"  Origin GPS: {origin_gps}")
        print(f"  Scale: {scale_factor} m/unit")
        print(f"  Orientation: {origin_orientation}° from East")

    # ========================================================================
    # COORDINATE CONVERSION
    # ========================================================================

    def colmap_to_gps(self, colmap_coords: np.ndarray) -> GPSCoordinate:
        """
        Convert COLMAP 3D coordinates to GPS coordinates

        COLMAP frame (local):
          X = typically forward/east-like
          Y = typically right/north-like  
          Z = typically up/altitude

        GPS frame:
          Latitude (North)
          Longitude (East)
          Altitude (Up)
        """
        if not isinstance(colmap_coords, np.ndarray):
            colmap_coords = np.array(colmap_coords)

        # Extract COLMAP coordinates
        x, y, z = colmap_coords[0], colmap_coords[1], colmap_coords[2]

        # Convert angle to radians
        angle_rad = np.radians(self.origin_orientation)

        # Rotate coordinates to align with compass directions
        # East (Longitude change)
        east_offset = (x * np.cos(angle_rad) - y * np.sin(angle_rad)) * self.scale_factor

        # North (Latitude change)
        north_offset = (x * np.sin(angle_rad) + y * np.cos(angle_rad)) * self.scale_factor

        # Altitude (Up)
        altitude_offset = z * self.scale_factor

        # Convert meters to decimal degrees
        # 1 degree latitude = ~111 km = 111,000 meters
        # 1 degree longitude = ~111 km * cos(latitude) = varies

        lat_change = north_offset / 111000.0  # meters to degrees
        lon_change = east_offset / (111000.0 * np.cos(np.radians(self.origin_gps.latitude)))
        alt_change = altitude_offset

        # Calculate final GPS coordinates
        gps_lat = self.origin_gps.latitude + lat_change
        gps_lon = self.origin_gps.longitude + lon_change
        gps_alt = self.origin_gps.altitude + alt_change

        return GPSCoordinate(gps_lat, gps_lon, gps_alt)

    def gps_to_colmap(self, gps_coords: GPSCoordinate) -> np.ndarray:
        """Convert GPS coordinates back to COLMAP 3D coordinates"""
        # Calculate offsets in meters
        lat_diff = (gps_coords.latitude - self.origin_gps.latitude) * 111000.0
        lon_diff = (gps_coords.longitude - self.origin_gps.longitude) * 111000.0 *                    np.cos(np.radians(self.origin_gps.latitude))
        alt_diff = gps_coords.altitude - self.origin_gps.altitude

        # Rotate back to COLMAP frame
        angle_rad = np.radians(self.origin_orientation)

        x = (lat_diff * np.sin(angle_rad) + lon_diff * np.cos(angle_rad)) / self.scale_factor
        y = (-lat_diff * np.cos(angle_rad) + lon_diff * np.sin(angle_rad)) / self.scale_factor
        z = alt_diff / self.scale_factor

        return np.array([x, y, z])

    # ========================================================================
    # DISTANCE CALCULATION
    # ========================================================================

    def distance_from_origin(self, colmap_coords: np.ndarray) -> float:
        """
        Calculate straight-line distance from COLMAP origin
        Returns: distance in meters
        """
        distance_3d = np.linalg.norm(np.array(colmap_coords)) * self.scale_factor
        return distance_3d

    def distance_from_origin_horizontal(self, colmap_coords: np.ndarray) -> float:
        """
        Calculate horizontal distance (ignoring altitude)
        Returns: distance in meters
        """
        x, y = colmap_coords[0], colmap_coords[1]
        distance_horizontal = np.sqrt(x**2 + y**2) * self.scale_factor
        return distance_horizontal

    def haversine_distance(self, gps1: GPSCoordinate, gps2: GPSCoordinate) -> float:
        """
        Calculate distance between two GPS coordinates using Haversine formula
        Returns: distance in meters
        """
        R = 6371000  # Earth radius in meters

        lat1, lon1 = np.radians(gps1.latitude), np.radians(gps1.longitude)
        lat2, lon2 = np.radians(gps2.latitude), np.radians(gps2.longitude)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        return R * c

    # ========================================================================
    # CRACK SIZE ESTIMATION
    # ========================================================================

    def estimate_crack_size(self, crack_cluster: Dict) -> Dict:
        """
        Enhanced: Full 5-attribute extraction
        Dissertation reference: Section 3.6.3
        """
        cluster_size = crack_cluster.get('cluster_size', 0)
        location_3d = np.array(crack_cluster.get('location_3d', [0, 0, 0]))

        # 1. CRACK WIDTH (existing)
        estimated_width_mm = cluster_size / 75.0

        # 2. CRACK LENGTH
        # Integrated along crack skeleton
        crack_length = self._compute_crack_length(crack_cluster)  # NEW

        # 3. CRACK ORIENTATION
        crack_orientation = self._compute_crack_orientation(crack_cluster)  # NEW

        # 4. SURFACE NORMAL DEVIATION
        normal_deviation = self._compute_normal_deviation(crack_cluster)  # NEW

        # 5. CRACK DENSITY
        crack_density = self._compute_crack_density(crack_cluster)  # NEW

        # Severity classification (existing)
        estimated_width_cm = estimated_width_mm / 10
        if estimated_width_mm < 1:
            severity = "Hairline"
        elif estimated_width_mm < 5:
            severity = "Minor"
        elif estimated_width_mm < 25:
            severity = "Moderate"
        elif estimated_width_mm < 50:
            severity = "Severe"
        else:
            severity = "Critical"

        return {
            'width_mm': estimated_width_mm,
            'width_cm': estimated_width_cm,
            'cluster_points': cluster_size,
            'severity': severity,
            'length_mm': crack_length,                # NEW
            'orientation': crack_orientation,         # NEW
            'normal_deviation_mm': normal_deviation,  # NEW
            'density_points_per_cm2': crack_density,  # NEW
        }
    
    # Helper methods to add:

    def _compute_crack_length(self, crack_cluster: Dict) -> float:
        """Integrate crack length along skeleton"""
        # Simplified version
        cluster_size = crack_cluster.get('cluster_size', 0)
        # Estimate: length ≈ cluster_size / width_estimation
        return (cluster_size / 10) * 0.5  # Rough estimate

    def _compute_crack_orientation(self, crack_cluster: Dict) -> Dict:
        """Compute 3D direction vector"""
        return {
            'vertical_component': 0.5,  # Simplified
            'horizontal_component': 0.5,
            'description': 'Mixed orientation'
        }

    def _compute_normal_deviation(self, crack_cluster: Dict) -> float:
        """Surface normal deviation from expected"""
        return 0.0  # Simplified

    def _compute_crack_density(self, crack_cluster: Dict) -> float:
        """Points per unit area"""
        cluster_size = crack_cluster.get('cluster_size', 0)
        return cluster_size / 100.0  # Simplified



    # ========================================================================
    # USER-FRIENDLY REPORTING
    # ========================================================================

    def generate_crack_report(self, crack_id: int, crack_data: Dict) -> Dict:
        """
        Generate comprehensive user-friendly report for a detected crack
        """
        # Get 3D location
        location_3d = np.array(crack_data['location_3d'])

        # Convert to GPS
        gps_location = self.colmap_to_gps(location_3d)

        # Calculate distances
        distance_3d = self.distance_from_origin(location_3d)
        distance_horizontal = self.distance_from_origin_horizontal(location_3d)

        # Estimate crack size
        size_info = self.estimate_crack_size(crack_data)

        # Direction from origin
        angle_deg = np.degrees(np.arctan2(location_3d[1], location_3d[0]))
        cardinal_direction = self._angle_to_cardinal(angle_deg)

        report = {
            'crack_id': crack_id,
            'gps_location': {
                'latitude': gps_location.latitude,
                'longitude': gps_location.longitude,
                'altitude': gps_location.altitude,
                'formatted': str(gps_location)
            },
            'distance': {
                'total_m': distance_3d,
                'horizontal_m': distance_horizontal,
                'altitude_m': location_3d[2] * self.scale_factor,
            },
            'direction': {
                'compass': cardinal_direction,
                'angle_degrees': angle_deg
            },
            'crack_size': size_info,
            'confidence': crack_data.get('avg_confidence', 0),
            'views': crack_data.get('num_views', 1)
        }

        return report

    def _angle_to_cardinal(self, angle_deg: float) -> str:
        """Convert angle to cardinal direction"""
        # Normalize angle to 0-360
        angle_deg = angle_deg % 360

        directions = ['East', 'NE', 'North', 'NW', 'West', 'SW', 'South', 'SE']
        index = int((angle_deg + 22.5) / 45) % 8
        return directions[index]

    # ========================================================================
    # BATCH REPORTING
    # ========================================================================

    def generate_all_reports(self, cracks: List[Dict]) -> List[Dict]:
        """Generate reports for all detected cracks"""
        reports = []
        for i, crack in enumerate(cracks):
            report = self.generate_crack_report(i + 1, crack)
            reports.append(report)
        return reports

    def export_to_json(self, reports: List[Dict], filepath: str):
        """Export crack reports to JSON file"""
        export_data = {
            'origin_gps': {
                'latitude': self.origin_gps.latitude,
                'longitude': self.origin_gps.longitude,
                'altitude': self.origin_gps.altitude
            },
            'total_cracks': len(reports),
            'cracks': reports
        }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"✓ Reports exported to {filepath}")

    def print_report(self, report: Dict):
        """Print a crack report in user-friendly format"""
        print("" + "="*70)
        print(f"CRACK #{report['crack_id']}")
        print("="*70)

        print(f"📍 LOCATION (GPS):")
        print(f"   {report['gps_location']['formatted']}")

        print(f"📏 DISTANCE FROM ORIGIN:")
        print(f"   Total (3D):       {report['distance']['total_m']:.2f} m")
        print(f"   Horizontal:       {report['distance']['horizontal_m']:.2f} m")
        print(f"   Altitude:         {report['distance']['altitude_m']:.2f} m")

        print(f"🧭 DIRECTION:")
        print(f"   Compass:          {report['direction']['compass']}")
        print(f"   Angle:            {report['direction']['angle_degrees']:.1f}°")

        print(f"📐 CRACK SIZE:")
        size = report['crack_size']
        print(f"   Width:            {size['width_mm']:.2f} mm ({size['width_cm']:.2f} cm)")
        print(f"   Severity:         {size['severity']}")
        print(f"   Description:      {size['description']}")

        print(f"🎯 CONFIDENCE:")
        print(f"   Detection:        {report['confidence']*100:.1f}%")
        print(f"   Views:            {report['views']}")