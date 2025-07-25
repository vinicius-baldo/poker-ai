"""
Game State Detector: Comprehensive detection of poker game state.
"""
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.poker_engine import Card
from vision.enhanced_card_recognizer import EnhancedCardRecognizer
from vision.ocr_detector import OCRDetector
from vision.position_detector import PositionDetector

logger = logging.getLogger(__name__)


@dataclass
class GameState:
    """Represents the current state of a poker game."""

    # Cards
    hole_cards: List[Card]
    community_cards: List[Card]

    # Money
    pot_size: float
    current_bet: float
    player_stack: float
    opponent_stack: float

    # Game info
    blind_level: Tuple[int, int]
    street: str  # preflop, flop, turn, river
    position: str  # UTG, BB, SB, etc.

    # Players
    active_players: int
    button_position: int

    # Detection confidence
    confidence: float


class GameStateDetector:
    """Comprehensive game state detection for PokerStars."""

    def __init__(self, config_path: str = "config/table_regions.json"):
        self.card_recognizer = EnhancedCardRecognizer()
        self.ocr_detector = OCRDetector()
        self.position_detector = PositionDetector()

        # Load configuration
        self.config = self._load_config(config_path)

        # Detection state
        self.last_detection: Optional[GameState] = None
        self.detection_confidence = 0.0

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load table regions configuration."""
        try:
            import json

            with open(config_path, "r") as f:
                config = json.load(f)
                return config if isinstance(config, dict) else {}
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
            return {}

    def detect_game_state(self, image: np.ndarray) -> Optional[GameState]:
        """
        Detect the complete game state from an image.

        Args:
            image: PokerStars table screenshot

        Returns:
            GameState object or None if detection fails
        """
        try:
            logger.info("Detecting game state...")

            # Detect cards
            hole_cards = self._detect_hole_cards(image)
            community_cards = self._detect_community_cards(image)

            # Detect money amounts
            pot_size = self._detect_pot_size(image)
            current_bet = self._detect_current_bet(image)
            player_stack = self._detect_player_stack(image)
            opponent_stack = self._detect_opponent_stack(image)

            # Detect game info
            blind_level = self._detect_blind_level(image)
            street = self._determine_street(community_cards)
            position = self._detect_position(image)

            # Detect players
            active_players = self._detect_active_players(image)
            button_position = self._detect_button_position(image)

            # Calculate confidence
            confidence = self._calculate_confidence(
                hole_cards, community_cards, pot_size, current_bet
            )

            # Create game state
            game_state = GameState(
                hole_cards=hole_cards or [],
                community_cards=community_cards or [],
                pot_size=pot_size or 0.0,
                current_bet=current_bet or 0.0,
                player_stack=player_stack or 100.0,
                opponent_stack=opponent_stack or 100.0,
                blind_level=blind_level or (1, 2),
                street=street,
                position=position or "unknown",
                active_players=active_players or 2,
                button_position=button_position or 0,
                confidence=confidence,
            )

            self.last_detection = game_state
            self.detection_confidence = confidence

            logger.info(f"Game state detected with {confidence:.1%} confidence")
            return game_state

        except Exception as e:
            logger.error(f"Game state detection failed: {e}")
            return None

    def _detect_hole_cards(self, image: np.ndarray) -> List[Card]:
        """Detect hero's hole cards."""
        try:
            # Get hole cards region from config
            region = self.config.get("poker_stars_regions", {}).get("hole_cards")
            if not region:
                logger.warning("Hole cards region not configured")
                return []

            x, y, w, h = region["x"], region["y"], region["width"], region["height"]
            hole_cards_region = image[y : y + h, x : x + w]

            # Use card recognizer
            cards: List[Card] = self.card_recognizer.recognize_hole_cards(
                hole_cards_region
            )

            logger.debug(f"Detected hole cards: {cards}")
            return cards

        except Exception as e:
            logger.error(f"Hole cards detection failed: {e}")
            return []

    def _detect_community_cards(self, image: np.ndarray) -> List[Card]:
        """Detect community cards (flop, turn, river)."""
        try:
            # Get community cards region from config
            region = self.config.get("poker_stars_regions", {}).get("community_cards")
            if not region:
                logger.warning("Community cards region not configured")
                return []

            x, y, w, h = region["x"], region["y"], region["width"], region["height"]
            community_cards_region = image[y : y + h, x : x + w]

            # Use card recognizer
            cards: List[Card] = self.card_recognizer.recognize_community_cards(
                community_cards_region
            )

            logger.debug(f"Detected community cards: {cards}")
            return cards

        except Exception as e:
            logger.error(f"Community cards detection failed: {e}")
            return []

    def _detect_pot_size(self, image: np.ndarray) -> Optional[float]:
        """Detect pot size using OCR."""
        try:
            # Get pot size region from config
            region = self.config.get("poker_stars_regions", {}).get("pot_size")
            if not region:
                logger.warning("Pot size region not configured")
                return None

            x, y, w, h = region["x"], region["y"], region["width"], region["height"]
            pot_region = image[y : y + h, x : x + w]

            # Use OCR detector
            pot_size: Optional[float] = self.ocr_detector.detect_pot_size(pot_region)

            logger.debug(f"Detected pot size: {pot_size}")
            return pot_size

        except Exception as e:
            logger.error(f"Pot size detection failed: {e}")
            return None

    def _detect_current_bet(self, image: np.ndarray) -> Optional[float]:
        """Detect current bet amount using OCR."""
        try:
            # Get current bet region from config
            region = self.config.get("poker_stars_regions", {}).get("current_bet")
            if not region:
                logger.warning("Current bet region not configured")
                return None

            x, y, w, h = region["x"], region["y"], region["width"], region["height"]
            bet_region = image[y : y + h, x : x + w]

            # Use OCR detector
            bet_amount: Optional[float] = self.ocr_detector.detect_bet_amount(
                bet_region
            )

            logger.debug(f"Detected current bet: {bet_amount}")
            return bet_amount

        except Exception as e:
            logger.error(f"Current bet detection failed: {e}")
            return None

    def _detect_player_stack(self, image: np.ndarray) -> Optional[float]:
        """Detect player stack size using OCR."""
        # TODO: Implement player stack detection
        return None

    def _detect_opponent_stack(self, image: np.ndarray) -> Optional[float]:
        """Detect opponent stack size using OCR."""
        # TODO: Implement opponent stack detection
        return None

    def _detect_blind_level(self, image: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect blind level using OCR."""
        try:
            # Get blind level region from config
            region = self.config.get("tournament_regions", {}).get("blind_level")
            if not region:
                logger.warning("Blind level region not configured")
                return None

            x, y, w, h = region["x"], region["y"], region["width"], region["height"]
            blind_region = image[y : y + h, x : x + w]

            # Use OCR detector
            blind_level: Optional[
                Tuple[int, int]
            ] = self.ocr_detector.detect_blind_level(blind_region)

            logger.debug(f"Detected blind level: {blind_level}")
            return blind_level

        except Exception as e:
            logger.error(f"Blind level detection failed: {e}")
            return None

    def _determine_street(self, community_cards: List[Card]) -> str:
        """Determine the current street based on community cards."""
        if not community_cards:
            return "preflop"
        elif len(community_cards) == 3:
            return "flop"
        elif len(community_cards) == 4:
            return "turn"
        elif len(community_cards) == 5:
            return "river"
        else:
            return "unknown"

    def _detect_position(self, image: np.ndarray) -> str:
        """Detect player position."""
        # TODO: Implement position detection
        return "unknown"

    def _detect_active_players(self, image: np.ndarray) -> int:
        """Detect number of active players."""
        # TODO: Implement active players detection
        return 2

    def _detect_button_position(self, image: np.ndarray) -> int:
        """Detect button position."""
        # TODO: Implement button position detection
        return 0

    def _calculate_confidence(
        self,
        hole_cards: List[Card],
        community_cards: List[Card],
        pot_size: Optional[float],
        current_bet: Optional[float],
    ) -> float:
        """Calculate detection confidence."""
        confidence = 0.0
        total_checks = 0

        # Card detection confidence
        if hole_cards:
            confidence += min(len(hole_cards) / 2.0, 1.0)  # Max 1.0 for 2 cards
            total_checks += 1

        if community_cards:
            confidence += min(len(community_cards) / 5.0, 1.0)  # Max 1.0 for 5 cards
            total_checks += 1

        # Money detection confidence
        if pot_size is not None:
            confidence += 1.0
            total_checks += 1

        if current_bet is not None:
            confidence += 1.0
            total_checks += 1

        return confidence / total_checks if total_checks > 0 else 0.0

    def get_last_detection(self) -> Optional[GameState]:
        """Get the last detected game state."""
        return self.last_detection

    def get_detection_confidence(self) -> float:
        """Get the confidence of the last detection."""
        return self.detection_confidence


def test_game_state_detector() -> None:
    """Test the game state detector."""
    print("🎯 Testing Game State Detector")
    print("=" * 40)

    detector = GameStateDetector()

    # Test with sample image if available
    test_image = "imagem_tela.png"
    if os.path.exists(test_image):
        print(f"Testing with: {test_image}")

        image = cv2.imread(test_image)
        if image is not None:
            game_state = detector.detect_game_state(image)

            if game_state:
                print("✅ Game state detected:")
                print(f"  Hole cards: {game_state.hole_cards}")
                print(f"  Community cards: {game_state.community_cards}")
                print(f"  Pot size: ${game_state.pot_size}")
                print(f"  Current bet: ${game_state.current_bet}")
                print(f"  Street: {game_state.street}")
                print(f"  Confidence: {game_state.confidence:.1%}")
            else:
                print("❌ Game state detection failed")
    else:
        print("No test image found. Create a screenshot to test detection.")


if __name__ == "__main__":
    import os

    test_game_state_detector()
