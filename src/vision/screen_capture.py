"""
Screen capture functionality for reading poker table information.
"""
import cv2
import numpy as np
from PIL import Image
import mss
import mss.tools
from typing import Tuple, Optional, Dict, List
import logging
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from number_reader import NumberReader

logger = logging.getLogger(__name__)


class ScreenCapture:
    """Handles screen capture for poker table reading."""
    
    def __init__(self):
        self.sct = mss.mss()
        self.regions: Dict[str, Dict] = {}
        self.last_capture: Optional[np.ndarray] = None
        self.number_reader = NumberReader()
    
    def add_region(self, name: str, x: int, y: int, width: int, height: int):
        """Add a screen region to capture."""
        self.regions[name] = {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "monitor": {"top": y, "left": x, "width": width, "height": height}
        }
        logger.info(f"Added capture region '{name}': {x},{y} {width}x{height}")
    
    def capture_region(self, region_name: str) -> Optional[np.ndarray]:
        """Capture a specific region of the screen."""
        if region_name not in self.regions:
            logger.error(f"Region '{region_name}' not found")
            return None
        
        try:
            # Capture the region
            screenshot = self.sct.grab(self.regions[region_name]["monitor"])
            
            # Convert to numpy array
            img = np.array(screenshot)
            
            # Convert from BGRA to BGR (OpenCV format)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            self.last_capture = img
            return img
            
        except Exception as e:
            logger.error(f"Failed to capture region '{region_name}': {e}")
            return None
    
    def capture_all_regions(self) -> Dict[str, np.ndarray]:
        """Capture all defined regions."""
        captures = {}
        for region_name in self.regions:
            capture = self.capture_region(region_name)
            if capture is not None:
                captures[region_name] = capture
        return captures
    
    def save_capture(self, image: np.ndarray, filename: str):
        """Save a captured image to file."""
        try:
            cv2.imwrite(filename, image)
            logger.info(f"Saved capture to {filename}")
        except Exception as e:
            logger.error(f"Failed to save capture: {e}")
    
    def get_region_info(self, region_name: str) -> Optional[Dict]:
        """Get information about a capture region."""
        return self.regions.get(region_name)

    def read_pot_size(self) -> Optional[float]:
        """Capture and read the pot size from the table."""
        img = self.capture_region("pot_size")
        if img is not None:
            return self.number_reader.read_number(img)
        return None

    def read_bet_amount(self) -> Optional[float]:
        """Capture and read the current bet amount from the table."""
        img = self.capture_region("bet_amounts")
        if img is not None:
            return self.number_reader.read_number(img)
        return None

    def read_player_stack(self, player_region_name: str) -> Optional[float]:
        """Capture and read a player's stack from a specific region."""
        img = self.capture_region(player_region_name)
        if img is not None:
            return self.number_reader.read_number(img)
        return None


class PokerStarsRegions:
    """Predefined regions for PokerStars table layout."""
    
    # These are example coordinates - will need calibration for different screen sizes
    REGIONS = {
        "hole_cards": {"x": 400, "y": 500, "width": 120, "height": 60},
        "community_cards": {"x": 350, "y": 300, "width": 220, "height": 60},
        "pot_size": {"x": 450, "y": 280, "width": 80, "height": 25},
        "player_names": {"x": 300, "y": 400, "width": 300, "height": 200},
        "action_buttons": {"x": 350, "y": 600, "width": 200, "height": 50},
        "bet_amounts": {"x": 350, "y": 580, "width": 200, "height": 20},
        "player_stacks": {"x": 300, "y": 450, "width": 300, "height": 150}
    }
    
    @staticmethod
    def setup_regions(capture: ScreenCapture):
        """Setup standard PokerStars regions."""
        for name, coords in PokerStarsRegions.REGIONS.items():
            capture.add_region(name, coords["x"], coords["y"], coords["width"], coords["height"])
    
    @staticmethod
    def calibrate_regions(capture: ScreenCapture, reference_image: str):
        """
        Calibrate regions based on a reference image.
        This would be used to adjust coordinates for different screen sizes.
        """
        # TODO: Implement automatic region calibration
        logger.info("Region calibration not yet implemented")
        pass


class ImageProcessor:
    """Processes captured images for better OCR results."""
    
    @staticmethod
    def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
    
    @staticmethod
    def enhance_card_image(image: np.ndarray) -> np.ndarray:
        """Enhance image specifically for card recognition."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Apply slight blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        return blurred
    
    @staticmethod
    def detect_text_regions(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect regions that likely contain text."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply morphological operations to find text regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter by size (text regions should be reasonably sized)
            if 20 < w < 200 and 10 < h < 50:
                text_regions.append((x, y, w, h))
        
        return text_regions 