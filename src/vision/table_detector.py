"""
Table Detector: Integrates vision components for poker table detection.
"""
import json
import logging
import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .card_recognizer import CardRecognizer
from .number_reader import NumberReader

logger = logging.getLogger(__name__)


class TableDetector:
    """Detects poker table information using vision components."""

    def __init__(self, config_path: str = "config/table_regions.json") -> None:
        """Initialize table detector with configuration."""
        self.config_path = config_path
        self.regions = self._load_regions()
        self.number_reader = NumberReader()
        self.card_recognizer = CardRecognizer()

    def _load_regions(self) -> Dict[str, Any]:
        """Load table regions from configuration file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r") as f:
                    config = json.load(f)
                    return config.get("regions", {})
            else:
                logger.warning(
                    f"Config file {self.config_path} not found, using defaults"
                )
                return self._get_default_regions()
        except Exception as e:
            logger.error(f"Error loading regions: {e}")
            return self._get_default_regions()

    def _get_default_regions(self) -> Dict[str, Any]:
        """Get default regions if config file is not available."""
        return {
            "pot_size": {"x": 450, "y": 280, "width": 80, "height": 25},
            "hole_cards": {"x": 400, "y": 500, "width": 120, "height": 60},
            "community_cards": {"x": 350, "y": 300, "width": 220, "height": 60},
            "current_bet": {"x": 350, "y": 580, "width": 200, "height": 20},
        }

    def detect_all(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Detect all table information from an image.

        Args:
            image_path: Path to the poker table image

        Returns:
            Dictionary with detected information
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return None

            # Detect pot size
            pot_size = self._detect_pot_size(image)

            # Detect hole cards
            hole_cards = self._detect_hole_cards(image)

            # Detect community cards
            community_cards = self._detect_community_cards(image)

            # Detect current bet
            current_bet = self._detect_current_bet(image)

            # Compile results
            results = {
                "pot_size": pot_size or 0.0,
                "hole_cards": hole_cards or [],
                "community_cards": community_cards or [],
                "current_bet": current_bet or 0.0,
                "detection_confidence": 0.8,  # Placeholder
            }

            logger.info(
                f"Detected: pot=${pot_size}, {len(hole_cards or [])} hole cards, "
                f"{len(community_cards or [])} community cards"
            )

            return results

        except Exception as e:
            logger.error(f"Error detecting table information: {e}")
            return None

    def _detect_pot_size(self, image: np.ndarray) -> Optional[float]:
        """Detect pot size from the table."""
        if "pot_size" not in self.regions:
            return None

        region = self.regions["pot_size"]
        x, y, w, h = region["x"], region["y"], region["width"], region["height"]

        # Extract region
        pot_region = image[y : y + h, x : x + w]

        # Read number from region
        pot_size = self.number_reader.read_number(pot_region)

        return pot_size

    def _detect_hole_cards(self, image: np.ndarray) -> Optional[List[str]]:
        """Detect hole cards from the table."""
        if "hole_cards" not in self.regions:
            return None

        region = self.regions["hole_cards"]
        x, y, w, h = region["x"], region["y"], region["width"], region["height"]

        # Extract region
        cards_region = image[y : y + h, x : x + w]

        # Recognize cards
        cards = self.card_recognizer.recognize_cards(cards_region)

        return cards

    def _detect_community_cards(self, image: np.ndarray) -> Optional[List[str]]:
        """Detect community cards from the table."""
        if "community_cards" not in self.regions:
            return None

        region = self.regions["community_cards"]
        x, y, w, h = region["x"], region["y"], region["width"], region["height"]

        # Extract region
        cards_region = image[y : y + h, x : x + w]

        # Recognize cards
        cards = self.card_recognizer.recognize_cards(cards_region)

        return cards

    def _detect_current_bet(self, image: np.ndarray) -> Optional[float]:
        """Detect current bet amount from the table."""
        if "current_bet" not in self.regions:
            return None

        region = self.regions["current_bet"]
        x, y, w, h = region["x"], region["y"], region["width"], region["height"]

        # Extract region
        bet_region = image[y : y + h, x : x + w]

        # Read number from region
        bet_amount = self.number_reader.read_number(bet_region)

        return bet_amount

    def save_processed_regions(
        self, image_path: str, output_dir: str = "processed"
    ) -> None:
        """Save processed regions for debugging."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return

            os.makedirs(output_dir, exist_ok=True)

            for region_name, region in self.regions.items():
                x, y, w, h = region["x"], region["y"], region["width"], region["height"]
                region_img = image[y : y + h, x : x + w]

                output_path = os.path.join(output_dir, f"{region_name}.png")
                cv2.imwrite(output_path, region_img)

            logger.info(f"Saved processed regions to {output_dir}")

        except Exception as e:
            logger.error(f"Error saving processed regions: {e}")


def test_table_detector() -> None:
    """Test the table detector functionality."""
    print("🎯 Testing Table Detector...")
    print("=" * 40)

    detector = TableDetector()

    # Test with sample image if available
    test_image = "imagem_tela.png"
    if os.path.exists(test_image):
        print(f"📸 Testing with {test_image}...")

        # Detect table information
        results = detector.detect_all(test_image)

        if results:
            print("✅ Detection Results:")
            print(f"  Pot Size: ${results.get('pot_size', 0)}")
            print(f"  Hole Cards: {results.get('hole_cards', [])}")
            print(f"  Community Cards: {results.get('community_cards', [])}")
            print(f"  Current Bet: ${results.get('current_bet', 0)}")

            # Save processed regions
            detector.save_processed_regions(test_image)
        else:
            print("❌ Detection failed")
    else:
        print("⚠️ No test image found")

    print("✅ Table detector test completed!")


if __name__ == "__main__":
    test_table_detector()
