"""
Template Preprocessor for Enhanced Card Recognition
Applies consistent preprocessing to templates to improve matching accuracy.
"""
import logging
import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class TemplatePreprocessor:
    """Preprocesses card templates to improve matching accuracy."""

    def __init__(self, target_size: Tuple[int, int] = (40, 60)):
        self.target_size = target_size
        self.preprocessing_steps = [
            "resize",
            "grayscale",
            "normalize",
            "enhance_contrast",
            "remove_noise",
        ]

    def preprocess_template(self, template: np.ndarray) -> np.ndarray:
        """
        Apply comprehensive preprocessing to a template.

        Args:
            template: Input template image

        Returns:
            Preprocessed template
        """
        processed = template.copy()

        # Step 1: Resize to target size
        processed = self._resize_image(processed, self.target_size)

        # Step 2: Convert to grayscale if needed
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

        # Step 3: Normalize brightness and contrast
        processed = self._normalize_image(processed)

        # Step 4: Enhance contrast
        processed = self._enhance_contrast(processed)

        # Step 5: Remove noise
        processed = self._remove_noise(processed)

        return processed

    def _resize_image(
        self, image: np.ndarray, target_size: Tuple[int, int]
    ) -> np.ndarray:
        """Resize image to target size while maintaining aspect ratio."""
        h, w = image.shape[:2]
        target_w, target_h = target_size

        # Calculate scaling factor to maintain aspect ratio
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Convert to grayscale if it's a color image
        if len(resized.shape) == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Create target-sized canvas with white background
        canvas = np.ones((target_h, target_w), dtype=np.uint8) * 255

        # Center the resized image on the canvas
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2

        canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

        return canvas

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image brightness and contrast."""
        # Apply histogram equalization
        normalized = cv2.equalizeHist(image)

        # Normalize to 0-255 range
        normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)

        return normalized

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast for better feature detection."""
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)

        return enhanced

    def _remove_noise(self, image: np.ndarray) -> np.ndarray:
        """Remove noise while preserving edges."""
        # Apply bilateral filter to preserve edges
        denoised = cv2.bilateralFilter(image, 9, 75, 75)

        return denoised

    def create_multiple_sizes(self, template: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Create multiple sizes of a template for better matching.

        Args:
            template: Original template image

        Returns:
            Dictionary of templates at different sizes
        """
        sizes = {
            "small": (30, 45),
            "medium": (40, 60),
            "large": (50, 75),
            "xlarge": (60, 90),
        }

        templates = {}
        for size_name, size in sizes.items():
            templates[size_name] = self.preprocess_template(template)
            # Update target size for next iteration
            self.target_size = size

        return templates

    def match_with_preprocessing(
        self, region: np.ndarray, template: np.ndarray
    ) -> float:
        """
        Match a region against a template with preprocessing.

        Args:
            region: Detected card region
            template: Card template

        Returns:
            Matching score
        """
        # Preprocess both region and template
        processed_region = self.preprocess_template(region)
        processed_template = self.preprocess_template(template)

        # Ensure same size
        if processed_region.shape != processed_template.shape:
            processed_template = cv2.resize(
                processed_template,
                (processed_region.shape[1], processed_region.shape[0]),
            )

        # Template matching
        result = cv2.matchTemplate(
            processed_region, processed_template, cv2.TM_CCOEFF_NORMED
        )
        score = np.max(result)

        return score

    def batch_preprocess_templates(self, template_dir: str, output_dir: str) -> None:
        """
        Preprocess all templates in a directory.

        Args:
            template_dir: Directory containing original templates
            output_dir: Directory to save preprocessed templates
        """
        os.makedirs(output_dir, exist_ok=True)

        for filename in os.listdir(template_dir):
            if filename.endswith((".png", ".jpg", ".jpeg")):
                template_path = os.path.join(template_dir, filename)
                template = cv2.imread(template_path)

                if template is not None:
                    # Preprocess template
                    processed = self.preprocess_template(template)

                    # Save preprocessed template
                    output_path = os.path.join(output_dir, f"processed_{filename}")
                    cv2.imwrite(output_path, processed)

                    logger.info(f"Preprocessed {filename}")

        logger.info(f"Preprocessed templates saved to {output_dir}")

    def create_adaptive_templates(
        self, template: np.ndarray, region_sizes: list
    ) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Create templates adapted to specific region sizes.

        Args:
            template: Original template
            region_sizes: List of (width, height) tuples for detected regions

        Returns:
            Dictionary mapping region sizes to adapted templates
        """
        adaptive_templates = {}

        for size in region_sizes:
            # Temporarily set target size
            original_target = self.target_size
            self.target_size = size

            # Preprocess template for this size
            adapted = self.preprocess_template(template)
            adaptive_templates[size] = adapted

            # Restore original target size
            self.target_size = original_target

        return adaptive_templates


def test_preprocessing():
    """Test the template preprocessing functionality."""

    # Load a sample template
    template_path = "data/card_templates/3c.png"
    if not os.path.exists(template_path):
        print(f"❌ Template not found: {template_path}")
        return

    template = cv2.imread(template_path)
    if template is None:
        print("❌ Could not load template")
        return

    print("🔧 Testing Template Preprocessing")
    print("=" * 40)

    # Initialize preprocessor
    preprocessor = TemplatePreprocessor(target_size=(40, 60))

    # Preprocess template
    processed = preprocessor.preprocess_template(template)

    # Save processed template
    output_path = "data/card_templates/processed_3c.png"
    cv2.imwrite(output_path, processed)

    print(f"✅ Original template size: {template.shape}")
    print(f"✅ Processed template size: {processed.shape}")
    print(f"💾 Processed template saved to: {output_path}")

    # Create multiple sizes
    multiple_sizes = preprocessor.create_multiple_sizes(template)
    print(f"📏 Created {len(multiple_sizes)} different sizes")

    for size_name, sized_template in multiple_sizes.items():
        size_path = f"data/card_templates/{size_name}_3c.png"
        cv2.imwrite(size_path, sized_template)
        print(f"    {size_name}: {sized_template.shape}")


if __name__ == "__main__":
    test_preprocessing()
