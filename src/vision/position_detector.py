"""
Position Detector: Identifies player positions on the poker table.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PositionDetector:
    """Detects player positions on a poker table."""

    # Standard 9-player table positions
    POSITIONS = {
        "UTG": "Under the Gun",
        "UTG+1": "Under the Gun +1",
        "MP": "Middle Position",
        "MP+1": "Middle Position +1",
        "CO": "Cutoff",
        "BTN": "Button",
        "SB": "Small Blind",
        "BB": "Big Blind",
        "HERO": "Hero (You)",
    }

    def __init__(self, table_layout: str = "9max") -> None:
        """
        Initialize position detector.

        Args:
            table_layout: Table layout ("6max", "9max", etc.)
        """
        self.table_layout = table_layout
        self.player_positions: Dict[str, str] = {}
        self.button_position: Optional[str] = None
        self.hero_position: Optional[str] = None

    def detect_positions_from_image(self, image: np.ndarray) -> Dict[str, str]:
        """
        Detect player positions from table image.

        Args:
            image: Poker table screenshot

        Returns:
            Dictionary mapping player names to positions
        """
        try:
            # Convert to grayscale for processing
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect player avatars/seats
            seats = self._detect_player_seats(gray)

            # Detect button position
            button_pos = self._detect_button_position(gray)

            # Detect hero position (usually bottom center)
            hero_pos = self._detect_hero_position(gray)

            # Map positions based on layout
            positions = self._map_positions(seats, button_pos, hero_pos)

            self.player_positions = positions
            return positions

        except Exception as e:
            logger.error(f"Error detecting positions: {e}")
            return {}

    def _detect_player_seats(self, gray_image: np.ndarray) -> List[Tuple[int, int]]:
        """Detect player seat positions using template matching or contour detection."""
        seats = []

        # Method 1: Look for circular seat indicators
        circles = cv2.HoughCircles(
            gray_image,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=20,
            maxRadius=50,
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for x, y, r in circles:
                seats.append((x, y))

        # Method 2: Look for player name regions (if circles not found)
        if not seats:
            seats = self._detect_seats_by_text_regions(gray_image)

        return seats

    def _detect_seats_by_text_regions(
        self, gray_image: np.ndarray
    ) -> List[Tuple[int, int]]:
        """Detect seats by looking for player name text regions."""
        seats = []

        # Apply edge detection
        edges = cv2.Canny(gray_image, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Filter by size (player name regions are typically small rectangles)
            if 30 < w < 150 and 10 < h < 30:
                # Check if it looks like a player name region
                if self._is_player_name_region(gray_image[y : y + h, x : x + w]):
                    seats.append((x + w // 2, y + h // 2))

        return seats

    def _is_player_name_region(self, region: np.ndarray) -> bool:
        """Check if a region looks like it contains a player name."""
        # Simple heuristic: check if region has text-like characteristics
        # This could be enhanced with OCR
        return True  # Placeholder

    def _detect_button_position(
        self, gray_image: np.ndarray
    ) -> Optional[Tuple[int, int]]:
        """Detect the dealer button position."""
        # Look for the dealer button (usually a "D" or circular marker)

        # Method 1: Template matching for "D" button
        button_pos = self._find_button_by_template(gray_image)

        # Method 2: Look for circular button marker
        if button_pos is None:
            button_pos = self._find_button_by_circle(gray_image)

        return button_pos

    def _find_button_by_template(
        self, gray_image: np.ndarray
    ) -> Optional[Tuple[int, int]]:
        """Find button using template matching."""
        # This would require a button template image
        # For now, return None
        return None

    def _find_button_by_circle(
        self, gray_image: np.ndarray
    ) -> Optional[Tuple[int, int]]:
        """Find button by looking for small circular markers."""
        circles = cv2.HoughCircles(
            gray_image,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=25,
            minRadius=5,
            maxRadius=15,
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # Return the first small circle found (likely the button)
            if len(circles) > 0:
                x, y, r = circles[0]
                return (x, y)

        return None

    def _detect_hero_position(
        self, gray_image: np.ndarray
    ) -> Optional[Tuple[int, int]]:
        """Detect hero position (usually bottom center of table)."""
        height, width = gray_image.shape

        # Hero is typically at the bottom center
        hero_x = width // 2
        hero_y = height - 50  # 50 pixels from bottom

        return (hero_x, hero_y)

    def _map_positions(
        self,
        seats: List[Tuple[int, int]],
        button_pos: Optional[Tuple[int, int]],
        hero_pos: Optional[Tuple[int, int]],
    ) -> Dict[str, str]:
        """Map detected seats to position names."""
        positions: Dict[str, str] = {}

        if not seats:
            return positions

        # Sort seats by angle from center to determine position order
        center_x = sum(x for x, y in seats) // len(seats)
        center_y = sum(y for x, y in seats) // len(seats)

        # Calculate angles and sort seats
        seat_angles = []
        for i, (x, y) in enumerate(seats):
            angle = np.arctan2(y - center_y, x - center_x)
            seat_angles.append((angle, i, (x, y)))

        seat_angles.sort(key=lambda x: x[0])

        # Map to positions based on table layout
        if self.table_layout == "9max":
            position_names = [
                "UTG",
                "UTG+1",
                "MP",
                "MP+1",
                "CO",
                "BTN",
                "SB",
                "BB",
                "HERO",
            ]
        elif self.table_layout == "6max":
            position_names = ["UTG", "MP", "CO", "BTN", "SB", "BB"]
        else:
            position_names = [f"POS_{i}" for i in range(len(seats))]

        # Map seats to positions
        for i, (angle, seat_idx, (x, y)) in enumerate(seat_angles):
            if i < len(position_names):
                positions[f"Player_{seat_idx}"] = position_names[i]

        # Mark hero position if detected
        if hero_pos:
            # Find closest seat to hero position
            min_dist = float("inf")
            hero_seat = None
            for seat_idx, (x, y) in enumerate(seats):
                dist = np.sqrt((x - hero_pos[0]) ** 2 + (y - hero_pos[1]) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    hero_seat = seat_idx

            if hero_seat is not None:
                positions[f"Player_{hero_seat}"] = "HERO"

        return positions

    def get_position_advantage(self, position: str) -> str:
        """Get position advantage assessment."""
        position_rank = {
            "BTN": 1,
            "CO": 2,
            "MP+1": 3,
            "MP": 4,
            "UTG+1": 5,
            "UTG": 6,
            "SB": 7,
            "BB": 8,
        }

        if position in position_rank:
            rank = position_rank[position]
            if rank <= 2:
                return "excellent"
            elif rank <= 4:
                return "good"
            elif rank <= 6:
                return "neutral"
            else:
                return "poor"

        return "unknown"

    def get_relative_position(self, hero_pos: str, opponent_pos: str) -> str:
        """Get relative position of opponent to hero."""
        position_order = ["UTG", "UTG+1", "MP", "MP+1", "CO", "BTN", "SB", "BB"]

        try:
            hero_idx = position_order.index(hero_pos)
            opp_idx = position_order.index(opponent_pos)

            if opp_idx < hero_idx:
                return "in_front"
            elif opp_idx > hero_idx:
                return "behind"
            else:
                return "same"
        except ValueError:
            return "unknown"

    def set_hero_position(self, position: str) -> None:
        """Manually set hero position."""
        self.hero_position = position

    def set_button_position(self, position: str) -> None:
        """Manually set button position."""
        self.button_position = position

    def get_position_stats(self) -> Dict[str, Any]:
        """Get position statistics and information."""
        return {
            "table_layout": self.table_layout,
            "total_players": len(self.player_positions),
            "hero_position": self.hero_position,
            "button_position": self.button_position,
            "positions": self.player_positions,
        }


def test_position_detector() -> None:
    """Test the position detector functionality."""
    print("🎯 Testing Position Detector...")
    print("=" * 40)

    detector = PositionDetector("9max")

    # Test position advantage
    positions = ["BTN", "CO", "MP", "UTG", "BB"]
    for pos in positions:
        advantage = detector.get_position_advantage(pos)
        print(f"Position {pos}: {advantage} advantage")

    # Test relative position
    hero_pos = "CO"
    opponent_pos = "UTG"
    relative = detector.get_relative_position(hero_pos, opponent_pos)
    print(f"Hero ({hero_pos}) vs Opponent ({opponent_pos}): {relative}")

    print("✅ Position detector test completed!")


if __name__ == "__main__":
    test_position_detector()
