"""
Image Quality Assessment Module
Implements dissertation Phase 1: Pre-Processing quality checks
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


class ImageQualityAssessor:
    """Assess image quality before processing (Dissertation: Section 3.3.1)"""

    def __init__(self, blur_threshold: float = 100.0, 
                 exposure_low: float = 20, exposure_high: float = 235):
        """
        Initialize quality assessment

        Args:
            blur_threshold: Laplacian variance threshold (>100 = sharp)
            exposure_low: Minimum brightness percentile (0-255)
            exposure_high: Maximum brightness percentile (0-255)
        """
        self.blur_threshold = blur_threshold
        self.exposure_low = exposure_low
        self.exposure_high = exposure_high

    def assess_blur(self, image: np.ndarray) -> Tuple[float, bool]:
        """
        Detect blur using Laplacian variance

        Dissertation reference: Section 3.3.1
        Sharp images have high Laplacian variance

        Returns: (blur_score, is_acceptable)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = laplacian.var()

        is_acceptable = blur_score > self.blur_threshold
        return blur_score, is_acceptable

    def assess_exposure(self, image: np.ndarray) -> Tuple[float, bool]:
        """
        Detect over/under-exposure

        Dissertation reference: Section 3.3.1
        Optimal exposure: histogram in middle range

        Returns: (exposure_score, is_acceptable)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

        # Check for over/under-exposure
        dark_pixels = np.sum(hist[:50])
        bright_pixels = np.sum(hist[200:])
        total_pixels = image.shape[0] * image.shape[1]

        # Ideal: middle tones dominate
        dark_ratio = dark_pixels / total_pixels
        bright_ratio = bright_pixels / total_pixels

        exposure_score = 1.0 - (dark_ratio + bright_ratio)  # 0-1 score
        is_acceptable = (dark_ratio < 0.1) and (bright_ratio < 0.1)

        return exposure_score, is_acceptable

    def assess_contrast(self, image: np.ndarray) -> Tuple[float, bool]:
        """
        Assess image contrast (important for feature detection)

        Low contrast = fewer detectable features for SfM

        Returns: (contrast_score, is_acceptable)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        contrast = gray.std()  # Standard deviation = contrast measure
        is_acceptable = contrast > 20  # Threshold for acceptable contrast

        return contrast, is_acceptable

    def assess_image(self, image_path: str) -> Dict:
        """Complete quality assessment for single image"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {
                    'file': Path(image_path).name,
                    'status': 'ERROR: Cannot read image',
                    'acceptable': False,
                    'quality_score': 0.0
                }

            blur_score, blur_ok = self.assess_blur(image)
            exposure_score, exposure_ok = self.assess_exposure(image)
            contrast_score, contrast_ok = self.assess_contrast(image)

            # Overall quality score (0-100)
            quality_score = (
                (blur_score / self.blur_threshold) * 0.4 +
                (exposure_score) * 0.3 +
                (contrast_score / 50) * 0.3
            ) * 100
            quality_score = min(100, max(0, quality_score))

            # Overall acceptable if all checks pass
            acceptable = blur_ok and exposure_ok and contrast_ok

            return {
                'file': Path(image_path).name,
                'blur_score': float(blur_score),
                'blur_acceptable': blur_ok,
                'exposure_score': float(exposure_score),
                'exposure_acceptable': exposure_ok,
                'contrast_score': float(contrast_score),
                'contrast_acceptable': contrast_ok,
                'quality_score': float(quality_score),
                'acceptable': acceptable,
                'status': 'OK' if acceptable else 'POOR QUALITY'
            }

        except Exception as e:
            return {
                'file': Path(image_path).name,
                'status': f'ERROR: {str(e)}',
                'acceptable': False,
                'quality_score': 0.0
            }

    def assess_batch(self, image_paths: List[str]) -> Dict:
        """Assess multiple images"""
        results = []
        for path in image_paths:
            results.append(self.assess_image(path))

        acceptable_count = sum(1 for r in results if r['acceptable'])

        summary = {
            'total_images': len(image_paths),
            'acceptable_images': acceptable_count,
            'acceptable_ratio': acceptable_count / len(image_paths) if image_paths else 0,
            'average_quality_score': np.mean([r['quality_score'] for r in results]),
            'images': results
        }

        return summary

    def print_report(self, assessment: Dict):
        """Pretty print assessment results"""
        print("\n" + "="*70)
        print("IMAGE QUALITY ASSESSMENT REPORT")
        print("="*70)

        if 'total_images' in assessment:  # Batch assessment
            print(f"\nTotal images: {assessment['total_images']}")
            print(f"Acceptable: {assessment['acceptable_images']}/{assessment['total_images']} ({assessment['acceptable_ratio']*100:.1f}%)")
            print(f"Average quality: {assessment['average_quality_score']:.1f}/100")

            print(f"\nDetailed Results:")
            print(f"{'File':<30} {'Quality':<10} {'Status':<20}")
            print("-"*60)
            for img in assessment['images']:
                status = "✓ PASS" if img['acceptable'] else "✗ FAIL"
                print(f"{img['file']:<30} {img['quality_score']:<10.1f} {status:<20}")

        else:  # Single image
            print(f"\nFile: {assessment['file']}")
            print(f"Quality Score: {assessment['quality_score']:.1f}/100")
            print(f"Status: {assessment['status']}")

            if 'blur_score' in assessment:
                print(f"\nBlur: {assessment['blur_score']:.1f} {'✓' if assessment['blur_acceptable'] else '✗'}")
                print(f"Exposure: {assessment['exposure_score']:.2f} {'✓' if assessment['exposure_acceptable'] else '✗'}")
                print(f"Contrast: {assessment['contrast_score']:.1f} {'✓' if assessment['contrast_acceptable'] else '✗'}")