"""
OCR Detector: Uses OCR to read numbers from PokerStars screenshots.
"""
import logging
import re
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class OCRDetector:
    """Uses OCR to detect numbers in PokerStars screenshots."""

    def __init__(self) -> None:
        self.tesseract_available = self._check_tesseract()
        if not self.tesseract_available:
            logger.warning(
                "Tesseract not available. Install with: brew install tesseract"
            )

    def _check_tesseract(self) -> bool:
        """Check if Tesseract OCR is available."""
        try:
            import pytesseract

            # Try to get version to test if it's working
            pytesseract.get_tesseract_version()
            return True
        except (ImportError, Exception):
            return False

    def preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR accuracy."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Resize for better OCR
        height, width = gray.shape
        if width < 100:
            scale = 3.0
            gray = cv2.resize(gray, (int(width * scale), int(height * scale)))

        # Apply threshold to get black text on white background
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Remove noise
        kernel: np.ndarray = np.ones((1, 1), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        result: np.ndarray = thresh.astype(np.uint8)
        return result

    def extract_number(
        self, image: np.ndarray, region: Optional[Tuple[int, int, int, int]] = None
    ) -> Optional[float]:
        """
        Extract a number from an image region.

        Args:
            image: Input image
            region: Optional region (x, y, width, height) to extract

        Returns:
            Extracted number or None if failed
        """
        if not self.tesseract_available:
            return None

        try:
            import pytesseract

            # Extract region if specified
            if region:
                x, y, w, h = region
                roi = image[y : y + h, x : x + w]
            else:
                roi = image

            # Preprocess for OCR
            processed = self.preprocess_for_ocr(roi)

            # OCR configuration for numbers only
            config = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.,$€£₽¥"

            # Extract text
            text = pytesseract.image_to_string(processed, config=config).strip()

            # Extract number from text
            number = self._parse_number_from_text(text)

            if number is not None:
                logger.debug(f"OCR extracted: '{text}' -> {number}")

            return number

        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return None

    def _parse_number_from_text(self, text: str) -> Optional[float]:
        """Parse a number from OCR text."""
        if not text:
            return None

        # Remove common OCR artifacts
        text = text.replace("O", "0").replace("o", "0")  # Common OCR mistake
        text = text.replace("l", "1").replace("I", "1")  # Common OCR mistake
        text = text.replace("S", "5").replace("s", "5")  # Common OCR mistake

        # Find all numbers in the text
        # Look for patterns like: $123.45, 123.45, 123, etc.
        patterns = [
            r"[\$€£₽¥]?(\d+\.?\d*)",  # Currency + number
            r"(\d+\.?\d*)",  # Just number
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                try:
                    # Take the first match
                    number_str = matches[0]
                    number = float(number_str)
                    return number
                except ValueError:
                    continue

        return None

    def detect_pot_size(
        self, image: np.ndarray, region: Optional[Tuple[int, int, int, int]] = None
    ) -> Optional[float]:
        """Detect pot size from image."""
        return self.extract_number(image, region)

    def detect_bet_amount(
        self, image: np.ndarray, region: Optional[Tuple[int, int, int, int]] = None
    ) -> Optional[float]:
        """Detect bet amount from image."""
        return self.extract_number(image, region)

    def detect_stack_size(
        self, image: np.ndarray, region: Optional[Tuple[int, int, int, int]] = None
    ) -> Optional[float]:
        """Detect stack size from image."""
        return self.extract_number(image, region)

    def detect_blind_level(
        self, image: np.ndarray, region: Optional[Tuple[int, int, int, int]] = None
    ) -> Optional[Tuple[int, int]]:
        """Detect blind level (small blind/big blind) from image."""
        if not self.tesseract_available:
            return None

        try:
            import pytesseract

            # Extract region if specified
            if region:
                x, y, w, h = region
                roi = image[y : y + h, x : x + w]
            else:
                roi = image

            # Preprocess for OCR
            processed = self.preprocess_for_ocr(roi)

            # OCR configuration for numbers and slashes
            config = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789/"

            # Extract text
            text = pytesseract.image_to_string(processed, config=config).strip()

            # Parse blind level (e.g., "25/50")
            match = re.search(r"(\d+)/(\d+)", text)
            if match:
                small_blind = int(match.group(1))
                big_blind = int(match.group(2))
                return (small_blind, big_blind)

            return None

        except Exception as e:
            logger.error(f"Blind level detection failed: {e}")
            return None


def test_ocr_detector() -> None:
    """Test the OCR detector."""
    print("🔍 Testing OCR Detector")
    print("=" * 40)

    detector = OCRDetector()

    if not detector.tesseract_available:
        print("❌ Tesseract not available")
        print("Install with: brew install tesseract")
        print("Then install Python package: pip install pytesseract")
        return

    print("✅ Tesseract available")

    # Test with a sample image if available
    test_image = "imagem_tela.png"
    if os.path.exists(test_image):
        print(f"Testing with: {test_image}")

        image = cv2.imread(test_image)
        if image is not None:
            # Test pot size detection
            pot_size = detector.detect_pot_size(image)
            print(f"Pot size detected: {pot_size}")

            # Test bet amount detection
            bet_amount = detector.detect_bet_amount(image)
            print(f"Bet amount detected: {bet_amount}")

            # Test blind level detection
            blind_level = detector.detect_blind_level(image)
            print(f"Blind level detected: {blind_level}")
    else:
        print("No test image found. Create a screenshot to test OCR.")


if __name__ == "__main__":
    import os

    test_ocr_detector()
