"""
NumberReader: Extracts numeric values from images using pytesseract OCR.
"""
import cv2
import numpy as np
import pytesseract
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class NumberReader:
    """Extracts numbers (pot, bets, stacks) from images using OCR."""

    def __init__(self, tesseract_config: Optional[str] = None):
        # Default config: only digits, no spaces, single line
        self.tesseract_config = tesseract_config or '--psm 7 -c tessedit_char_whitelist=0123456789.'

    @staticmethod
    def preprocess(image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR accuracy (grayscale, threshold, resize)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Resize to improve OCR accuracy
        scale = 2
        resized = cv2.resize(gray, (image.shape[1]*scale, image.shape[0]*scale), interpolation=cv2.INTER_CUBIC)
        # Binarize
        _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def read_number(self, image: np.ndarray) -> Optional[float]:
        """Extract a numeric value from the image using OCR."""
        processed = self.preprocess(image)
        text = pytesseract.image_to_string(processed, config=self.tesseract_config)
        logger.debug(f"OCR raw output: '{text}'")
        # Clean and extract number
        text = text.replace(',', '').replace(' ', '').strip()
        try:
            value = float(text)
            logger.info(f"Extracted number: {value}")
            return value
        except ValueError:
            logger.warning(f"Failed to extract number from OCR text: '{text}'")
            return None

    def read_number_from_region(self, image: np.ndarray, region: tuple) -> Optional[float]:
        """Extract number from a specific region (x, y, w, h) of the image."""
        x, y, w, h = region
        cropped = image[y:y+h, x:x+w]
        return self.read_number(cropped) 