"""
Gentle Template Preprocessor for Enhanced Card Recognition
Applies minimal preprocessing to preserve card details while improving matching.
"""
import logging
import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class GentleTemplatePreprocessor:
    """Preprocesses card templates with minimal processing to preserve details."""

    def __init__(self, target_size: Tuple[int, int] = (40, 60)):
        self.target_size = target_size

    def preprocess_template(self, template: np.ndarray) -> np.ndarray:
        """
        Apply gentle preprocessing to preserve card details.

        Args:
            template: Input template image

        Returns:
            Preprocessed template
        """
        processed = template.copy()

        # Step 1: Convert to grayscale if needed
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

        # Step 2: Resize with high-quality interpolation
        processed = self._resize_high_quality(processed, self.target_size)

        # Step 3: Light normalization (preserve contrast)
        processed = self._light_normalize(processed)

        return processed

    def _resize_high_quality(
        self, image: np.ndarray, target_size: Tuple[int, int]
    ) -> np.ndarray:
        """Resize image with high-quality interpolation to preserve details."""
        h, w = image.shape[:2]
        target_w, target_h = target_size

        # Calculate scaling factor to maintain aspect ratio
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Use high-quality interpolation
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # Create target-sized canvas with white background
        canvas = np.ones((target_h, target_w), dtype=np.uint8) * 255

        # Center the resized image on the canvas
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2

        canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

        return canvas

    def _light_normalize(self, image: np.ndarray) -> np.ndarray:
        """Apply light normalization while preserving contrast."""
        # Simple normalization to 0-255 range
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

        # Optional: Light histogram equalization (only if needed)
        # normalized = cv2.equalizeHist(normalized)

        return normalized

    def create_multiple_sizes(self, template: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Create multiple sizes of a template with gentle processing.

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
            # Temporarily set target size
            original_target = self.target_size
            self.target_size = size

            templates[size_name] = self.preprocess_template(template)

            # Restore original target size
            self.target_size = original_target

        return templates

    def match_with_preprocessing(
        self, region: np.ndarray, template: np.ndarray
    ) -> float:
        """
        Match a region against a template with gentle preprocessing.

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
                interpolation=cv2.INTER_CUBIC,
            )

        # Template matching
        result = cv2.matchTemplate(
            processed_region, processed_template, cv2.TM_CCOEFF_NORMED
        )
        score = np.max(result)

        return score

    def batch_preprocess_templates(self, template_dir: str, output_dir: str) -> None:
        """
        Preprocess all templates in a directory with gentle processing.

        Args:
            template_dir: Directory containing original templates
            output_dir: Directory to save preprocessed templates
        """
        os.makedirs(output_dir, exist_ok=True)

        for filename in os.listdir(template_dir):
            if filename.endswith((".png", ".jpg", ".jpeg")) and not filename.startswith(
                "processed_"
            ):
                template_path = os.path.join(template_dir, filename)
                template = cv2.imread(template_path)

                if template is not None:
                    # Preprocess template
                    processed = self.preprocess_template(template)

                    # Save preprocessed template
                    output_path = os.path.join(output_dir, f"gentle_{filename}")
                    cv2.imwrite(output_path, processed)

                    logger.info(f"Gently preprocessed {filename}")

        logger.info(f"Gently preprocessed templates saved to {output_dir}")

    def create_adaptive_templates(
        self, template: np.ndarray, region_sizes: list
    ) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Create templates adapted to specific region sizes with gentle processing.

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


def test_gentle_preprocessing():
    """Test the gentle template preprocessing functionality."""

    # Load a sample template
    template_path = "data/card_templates/3c.png"
    if not os.path.exists(template_path):
        print(f"❌ Template not found: {template_path}")
        return

    template = cv2.imread(template_path)
    if template is None:
        print("❌ Could not load template")
        return

    print("🔧 Testing Gentle Template Preprocessing")
    print("=" * 45)

    # Initialize gentle preprocessor
    preprocessor = GentleTemplatePreprocessor(target_size=(40, 60))

    # Preprocess template
    processed = preprocessor.preprocess_template(template)

    # Save processed template
    output_path = "data/card_templates/gentle_3c.png"
    cv2.imwrite(output_path, processed)

    print(f"✅ Original template size: {template.shape}")
    print(f"✅ Gently processed template size: {processed.shape}")
    print(f"💾 Gently processed template saved to: {output_path}")

    # Create multiple sizes
    multiple_sizes = preprocessor.create_multiple_sizes(template)
    print(f"📏 Created {len(multiple_sizes)} different sizes")

    for size_name, sized_template in multiple_sizes.items():
        size_path = f"data/card_templates/gentle_{size_name}_3c.png"
        cv2.imwrite(size_path, sized_template)
        print(f"    {size_name}: {sized_template.shape}")


if __name__ == "__main__":
    test_gentle_preprocessing()
