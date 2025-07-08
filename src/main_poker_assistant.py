#!/usr/bin/env python3
"""
Main Poker Assistant: Integrates vision detection with AI analysis.
"""
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import pandas as pd

from ai.poker_advisor import PokerAdvisor, PokerAnalysisResult
from analysis.hand_tracker import HandTracker
from core.poker_engine import Card, Rank, Suit
from vision.position_detector import PositionDetector
from vision.screen_capture import PokerTableCapture
from vision.table_detector import TableDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PokerAssistant:
    """Main poker assistant that combines vision and AI analysis."""

    def __init__(
        self,
        config_path: str = "config/table_regions.json",
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize the poker assistant."""
        # Use standard table detector (LLM approach for advanced features)
        self.table_detector = TableDetector(config_path)
        logger.info("Initialized poker assistant")

        # Get API key from environment or parameter
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning(
                "No OpenAI API key provided. AI analysis will use fallback mode."
            )
            self.api_key = "dummy-key"

        self.advisor = PokerAdvisor(api_key=self.api_key)

        # Initialize components
        self.screen_capture = PokerTableCapture()
        self.position_detector = PositionDetector()
        self.hand_tracker = HandTracker()

        # Game state tracking
        self.current_street = "unknown"
        self.action_history: List[Dict[str, Any]] = []
        self.is_monitoring = False

    def analyze_current_situation(
        self, image_path: str
    ) -> Optional[PokerAnalysisResult]:
        """
        Analyze the current poker situation from an image.

        Args:
            image_path: Path to the poker table screenshot

        Returns:
            PokerAnalysisResult with AI recommendation, or None if detection fails
        """
        try:
            logger.info(f"Analyzing poker situation from image: {image_path}")

            # Detect table information
            table_info = self.table_detector.detect_all(image_path)

            if not table_info:
                logger.error("Failed to detect table information")
                return None

            # Convert detected cards to Card objects
            hole_cards = self._parse_cards(table_info.get("hole_cards", []))
            community_cards = self._parse_cards(table_info.get("community_cards", []))

            # Extract other information
            pot_size = table_info.get("pot_size", 0.0)
            current_bet = table_info.get("current_bet", 0.0)
            player_stack = table_info.get("player_stack", 100.0)  # Default value
            opponent_stack = table_info.get("opponent_stack", 100.0)  # Default value

            # Determine street based on community cards
            street = self._determine_street(community_cards)

            # Get position (this would need to be detected or configured)
            position = "unknown"  # Could be enhanced with position detection

            logger.info(
                f"Detected: {len(hole_cards)} hole cards, {len(community_cards)} community cards"
            )
            logger.info(f"Pot: ${pot_size}, Bet: ${current_bet}, Street: {street}")

            # Get AI analysis
            analysis = self.advisor.analyze_situation(
                hole_cards=hole_cards,
                community_cards=community_cards,
                pot_size=pot_size,
                current_bet=current_bet,
                player_stack=player_stack,
                opponent_stack=opponent_stack,
                position=position,
                street=street,
                action_history=self.action_history,
            )

            result = PokerAnalysisResult(analysis)

            # Log the recommendation
            logger.info(
                f"AI Recommendation: {result.recommendation.upper()} "
                f"(Confidence: {result.confidence:.1%})"
            )

            return result

        except Exception as e:
            logger.error(f"Error analyzing situation: {e}")
            return None

    def _parse_cards(self, card_strings: List[str]) -> List[Card]:
        """Parse card strings into Card objects."""
        cards = []

        for card_str in card_strings:
            try:
                # Handle different card formats
                if len(card_str) >= 2:
                    rank_char = card_str[0].upper()
                    suit_char = card_str[1].upper()

                    # Parse rank
                    rank_map = {
                        "A": Rank.ACE,
                        "K": Rank.KING,
                        "Q": Rank.QUEEN,
                        "J": Rank.JACK,
                        "T": Rank.TEN,
                        "9": Rank.NINE,
                        "8": Rank.EIGHT,
                        "7": Rank.SEVEN,
                        "6": Rank.SIX,
                        "5": Rank.FIVE,
                        "4": Rank.FOUR,
                        "3": Rank.THREE,
                        "2": Rank.TWO,
                    }

                    # Parse suit
                    suit_map = {
                        "H": Suit.HEARTS,
                        "D": Suit.DIAMONDS,
                        "C": Suit.CLUBS,
                        "S": Suit.SPADES,
                    }

                    if rank_char in rank_map and suit_char in suit_map:
                        cards.append(Card(rank_map[rank_char], suit_map[suit_char]))

            except Exception as e:
                logger.warning(f"Could not parse card '{card_str}': {e}")
                continue

        return cards

    def _determine_street(self, community_cards: List[Card]) -> str:
        """Determine the current street based on number of community cards."""
        num_cards = len(community_cards)

        if num_cards == 0:
            return "preflop"
        elif num_cards == 3:
            return "flop"
        elif num_cards == 4:
            return "turn"
        elif num_cards == 5:
            return "river"
        else:
            return "unknown"

    def add_action_to_history(
        self, player: str, action: str, amount: float = 0.0
    ) -> None:
        """Add an action to the history for context."""
        self.action_history.append(
            {"player": player, "action": action, "amount": amount}
        )

        # Keep only last 10 actions
        if len(self.action_history) > 10:
            self.action_history = self.action_history[-10:]

        # Also record in hand tracker
        self.hand_tracker.record_action(player, action, amount, self.current_street)

    def get_gui_data(self, image_path: str) -> Dict[str, Any]:
        """
        Get complete data for GUI display.

        Returns:
            Dict with all information needed for GUI:
            {
                "table_info": {...},
                "ai_analysis": {...},
                "cards": {...},
                "metrics": {...}
            }
        """
        # Get AI analysis
        analysis_result = self.analyze_current_situation(image_path)

        if not analysis_result:
            return {
                "error": "Failed to analyze situation",
                "table_info": {},
                "ai_analysis": {},
                "cards": {},
                "metrics": {},
            }

        # Get table detection info
        table_info = self.table_detector.detect_all(image_path) or {}

        # Prepare GUI data
        gui_data = {
            "table_info": {
                "pot_size": table_info.get("pot_size", 0.0),
                "current_bet": table_info.get("current_bet", 0.0),
                "street": self._determine_street(
                    self._parse_cards(table_info.get("community_cards", []))
                ),
            },
            "ai_analysis": analysis_result.to_gui_format(),
            "cards": {
                "hole_cards": table_info.get("hole_cards", []),
                "community_cards": table_info.get("community_cards", []),
            },
            "metrics": {
                "detection_confidence": table_info.get("detection_confidence", 0.0),
                "ai_confidence": analysis_result.confidence,
                "action_color": analysis_result.get_action_color(),
                "confidence_color": analysis_result.get_confidence_color(),
            },
            "raw_detection": table_info,  # For debugging
        }

        return gui_data

    def start_real_time_monitoring(
        self, table_region: Optional[Tuple[int, int, int, int]] = None
    ) -> None:
        """
        Start real-time monitoring of poker table.

        Args:
            table_region: Optional region to capture (x, y, width, height)
        """
        if self.is_monitoring:
            logger.warning("Already monitoring")
            return

        if table_region:
            self.screen_capture.set_table_region(table_region)

        def analysis_callback(image):
            """Callback for each captured image."""
            try:
                # Save temporary image
                temp_path = "temp_capture.png"
                cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

                # Analyze the situation
                result = self.analyze_current_situation(temp_path)

                if result:
                    logger.info(
                        f"Real-time analysis: {result.recommendation.upper()} "
                        f"({result.confidence:.1%})"
                    )

                    # Here you could trigger GUI updates, notifications, etc.

                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            except Exception as e:
                logger.error(f"Error in real-time analysis: {e}")

        # Start continuous capture
        self.screen_capture.start_continuous_capture(analysis_callback, interval=2.0)
        self.is_monitoring = True

        logger.info("Started real-time monitoring")

    def stop_real_time_monitoring(self) -> None:
        """Stop real-time monitoring."""
        if not self.is_monitoring:
            return

        self.screen_capture.stop_continuous_capture()
        self.is_monitoring = False
        logger.info("Stopped real-time monitoring")

    def calibrate_table_region(self) -> Optional[Tuple[int, int, int, int]]:
        """Calibrate the table region for capture."""
        return self.screen_capture.calibrate_table_region()

    def start_new_hand(self, hand_id: str = None) -> None:
        """Start tracking a new hand."""
        if hand_id is None:
            hand_id = f"hand_{int(time.time())}"

        table_info = {
            "table_name": "PokerStars",
            "stakes": "1/2",  # Could be detected
            "timestamp": time.time(),
        }

        self.hand_tracker.start_new_hand(hand_id, table_info)
        logger.info(f"Started new hand: {hand_id}")

    def end_current_hand(self) -> None:
        """End the current hand and update profiles."""
        self.hand_tracker.end_hand()
        logger.info("Ended current hand")

    def get_opponent_analysis(self, player_name: str) -> Dict[str, Any]:
        """Get detailed analysis of an opponent."""
        return self.hand_tracker.get_opponent_stats(player_name)

    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics."""
        return self.hand_tracker.get_session_stats()

    def close(self) -> None:
        """Close the poker assistant and clean up resources."""
        self.stop_real_time_monitoring()
        self.hand_tracker.close()
        self.screen_capture.close()
        logger.info("Poker assistant closed")

    def save_analysis_log(
        self, image_path: str, output_path: str = "analysis_log.json"
    ) -> None:
        """Save the current analysis to a log file."""
        try:
            gui_data = self.get_gui_data(image_path)

            # Add timestamp and image info
            log_entry = {
                "timestamp": str(pd.Timestamp.now()),
                "image_path": image_path,
                "analysis": gui_data,
            }

            # Load existing log or create new
            if os.path.exists(output_path):
                with open(output_path, "r") as f:
                    log = json.load(f)
            else:
                log = []

            log.append(log_entry)

            # Save updated log
            with open(output_path, "w") as f:
                json.dump(log, f, indent=2)

            logger.info(f"Analysis logged to {output_path}")

        except Exception as e:
            logger.error(f"Error saving analysis log: {e}")


def main() -> None:
    """Main function for testing the poker assistant."""
    if len(sys.argv) < 2:
        print("Usage: python main_poker_assistant.py <image_path>")
        print("Example: python main_poker_assistant.py imagem_tela.png")
        return

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found")
        return

    # Initialize assistant
    assistant = PokerAssistant()

    # Analyze situation
    print(f"🔍 Analyzing poker situation from: {image_path}")
    print("=" * 60)

    result = assistant.analyze_current_situation(image_path)

    if result:
        print(f"\n🎯 AI Recommendation: {result.recommendation.upper()}")
        print(f"📈 Confidence: {result.confidence:.1%}")
        print(f"💭 Reasoning: {result.reasoning}")
        print(f"💰 Expected Value: ${result.expected_value:.2f}")
        print(f"⚠️  Risk Level: {result.risk_level}")
        print(f"🃏 Hand Strength: {result.hand_strength}")
        print(f"📊 Pot Odds: {result.pot_odds:.2f}:1")

        # Show GUI format
        print(f"\n🖥️  GUI Data:")
        gui_data = result.to_gui_format()
        print(json.dumps(gui_data, indent=2))

        # Save analysis log
        assistant.save_analysis_log(image_path)

    else:
        print("❌ Failed to analyze situation")


if __name__ == "__main__":
    main()
