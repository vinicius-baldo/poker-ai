"""
Hand Tracker: Integrates opponent profiling with the main poker assistant.
"""
import json
import logging
import os
from typing import Any, Dict, List, Optional

from profiling.database import OpponentDatabase
from profiling.opponent_profile import OpponentProfile

logger = logging.getLogger(__name__)


class HandTracker:
    """Tracks hands and integrates opponent profiling with the main system."""

    def __init__(self, db_path: str = "data/opponents.db") -> None:
        """Initialize hand tracker with database."""
        self.database = OpponentDatabase(db_path)
        self.current_hand_id = None
        self.current_hand_data: Dict[str, Any] = {}
        self.action_history: List[Dict[str, Any]] = []
        self.opponents_in_hand: List[str] = []

    def start_new_hand(self, hand_id: str, table_info: Dict[str, Any]) -> None:
        """Start tracking a new hand."""
        self.current_hand_id = hand_id
        self.current_hand_data = {
            "hand_id": hand_id,
            "table_info": table_info,
            "actions": [],
            "players": [],
            "start_time": None,
            "end_time": None,
        }
        self.action_history = []
        self.opponents_in_hand = []

        logger.info(f"Started tracking hand {hand_id}")

    def add_player_to_hand(self, player_name: str, position: str, stack: float) -> None:
        """Add a player to the current hand."""
        player_data = {
            "name": player_name,
            "position": position,
            "stack": stack,
            "actions": [],
        }

        self.current_hand_data["players"].append(player_data)
        self.opponents_in_hand.append(player_name)

        # Get or create opponent profile
        profile = self.database.get_profile(player_name)
        logger.info(f"Added player {player_name} ({position}) to hand")

    def record_action(
        self,
        player_name: str,
        action: str,
        amount: float = 0.0,
        street: str = "unknown",
    ) -> None:
        """Record an action in the current hand."""
        action_data = {
            "player": player_name,
            "action": action,
            "amount": amount,
            "street": street,
            "timestamp": None,  # Could add timestamp if needed
        }

        self.action_history.append(action_data)
        self.current_hand_data["actions"].append(action_data)

        # Update player's actions in hand
        for player in self.current_hand_data["players"]:
            if player["name"] == player_name:
                player["actions"].append(action_data)
                break

        logger.debug(f"Recorded action: {player_name} {action} ${amount} on {street}")

    def end_hand(self, showdown_hands: Optional[Dict[str, List[str]]] = None) -> None:
        """End the current hand and update opponent profiles."""
        if not self.current_hand_id:
            logger.warning("No active hand to end")
            return

        # Update all opponent profiles
        for player_name in self.opponents_in_hand:
            if player_name == "Hero":  # Skip hero
                continue

            profile = self.database.get_profile(player_name)

            # Get player's actions for this hand
            player_actions = []
            for action in self.action_history:
                if action["player"] == player_name:
                    player_actions.append(action)

            # Determine if player went to showdown
            went_to_showdown = False
            showdown_hand = None

            if showdown_hands and player_name in showdown_hands:
                went_to_showdown = True
                showdown_hand = showdown_hands[player_name]

            # Update profile
            self.database.update_profile(
                name=player_name,
                actions=player_actions,
                went_to_showdown=went_to_showdown,
                showdown_hand=showdown_hand,
            )

        # Save hand data
        self._save_hand_data()

        logger.info(f"Ended hand {self.current_hand_id}")
        self.current_hand_id = None
        self.current_hand_data = {}
        self.action_history = []
        self.opponents_in_hand = []

    def get_opponent_stats(self, player_name: str) -> Dict[str, Any]:
        """Get opponent statistics for analysis."""
        profile = self.database.get_profile(player_name)
        stats = profile.get_stats()

        # Add additional analysis
        stats.update(self._analyze_opponent_tendencies(profile))

        return stats

    def get_opponents_in_hand(self) -> List[Dict[str, Any]]:
        """Get information about all opponents in the current hand."""
        opponents = []

        for player_name in self.opponents_in_hand:
            if player_name != "Hero":
                stats = self.get_opponent_stats(player_name)
                opponents.append({"name": player_name, "stats": stats})

        return opponents

    def get_hand_context_for_ai(self) -> Dict[str, Any]:
        """Get hand context information for AI analysis."""
        context = {
            "hand_id": self.current_hand_id,
            "players": len(self.opponents_in_hand),
            "action_history": self.action_history[-10:],  # Last 10 actions
            "opponents": self.get_opponents_in_hand(),
        }

        return context

    def _analyze_opponent_tendencies(self, profile: OpponentProfile) -> Dict[str, Any]:
        """Analyze opponent tendencies for AI recommendations."""
        stats = profile.get_stats()

        # Analyze playing style
        vpip = stats.get("vpip_pct", 0)
        pfr = stats.get("pfr_pct", 0)
        aggression = stats.get("aggression_factor", 0)

        # Determine playing style
        if vpip < 20:
            style = "tight"
        elif vpip < 35:
            style = "medium"
        else:
            style = "loose"

        # Determine aggression level
        if aggression > 2.0:
            aggression_level = "very_aggressive"
        elif aggression > 1.5:
            aggression_level = "aggressive"
        elif aggression > 0.8:
            aggression_level = "passive"
        else:
            aggression_level = "very_passive"

        # Calculate steal frequency (if we have position data)
        steal_frequency = 0.0  # Would need position data to calculate

        return {
            "playing_style": style,
            "aggression_level": aggression_level,
            "steal_frequency": steal_frequency,
            "is_regular": profile.hands_played > 50,
            "confidence": min(
                profile.hands_played / 100.0, 1.0
            ),  # Confidence based on sample size
        }

    def _save_hand_data(self) -> None:
        """Save hand data to file for analysis."""
        if not self.current_hand_data:
            return

        # Create data directory if it doesn't exist
        os.makedirs("data/hands", exist_ok=True)

        # Save hand data
        filename = f"data/hands/hand_{self.current_hand_id}.json"
        try:
            with open(filename, "w") as f:
                json.dump(self.current_hand_data, f, indent=2)
            logger.debug(f"Saved hand data to {filename}")
        except Exception as e:
            logger.error(f"Error saving hand data: {e}")

    def get_recent_hands(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent hand data for analysis."""
        hands = []
        data_dir = "data/hands"

        if not os.path.exists(data_dir):
            return hands

        # Get list of hand files
        hand_files = [
            f
            for f in os.listdir(data_dir)
            if f.startswith("hand_") and f.endswith(".json")
        ]
        hand_files.sort(reverse=True)  # Most recent first

        # Load recent hands
        for filename in hand_files[:limit]:
            try:
                with open(os.path.join(data_dir, filename), "r") as f:
                    hand_data = json.load(f)
                    hands.append(hand_data)
            except Exception as e:
                logger.error(f"Error loading hand data from {filename}: {e}")

        return hands

    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics for the current session."""
        recent_hands = self.get_recent_hands(50)  # Last 50 hands

        if not recent_hands:
            return {}

        total_hands = len(recent_hands)
        total_players = sum(len(hand.get("players", [])) for hand in recent_hands)

        # Calculate session statistics
        session_stats = {
            "total_hands": total_hands,
            "avg_players_per_hand": total_players / total_hands
            if total_hands > 0
            else 0,
            "unique_opponents": len(
                set(
                    player["name"]
                    for hand in recent_hands
                    for player in hand.get("players", [])
                    if player["name"] != "Hero"
                )
            ),
            "session_duration": None,  # Could calculate if we track timestamps
        }

        return session_stats

    def close(self) -> None:
        """Close the hand tracker and database."""
        self.database.close()


class HandAnalyzer:
    """Analyzes hand data for patterns and insights."""

    def __init__(self, hand_tracker: HandTracker) -> None:
        """Initialize hand analyzer."""
        self.hand_tracker = hand_tracker

    def analyze_opponent_range(
        self, player_name: str, street: str = "preflop"
    ) -> Dict[str, Any]:
        """Analyze opponent's likely hand range based on actions."""
        profile = self.hand_tracker.database.get_profile(player_name)
        stats = profile.get_stats()

        # Get recent actions for this player
        recent_actions = []
        for action in self.hand_tracker.action_history:
            if action["player"] == player_name and action["street"] == street:
                recent_actions.append(action)

        # Analyze betting patterns
        betting_analysis = self._analyze_betting_patterns(recent_actions)

        # Estimate hand range based on stats and actions
        estimated_range = self._estimate_hand_range(stats, betting_analysis)

        return {
            "player": player_name,
            "street": street,
            "estimated_range": estimated_range,
            "betting_patterns": betting_analysis,
            "confidence": stats.get("confidence", 0.0),
        }

    def _analyze_betting_patterns(
        self, actions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze betting patterns from actions."""
        if not actions:
            return {}

        bet_sizes = [action["amount"] for action in actions if action["amount"] > 0]
        action_types = [action["action"] for action in actions]

        analysis = {
            "total_actions": len(actions),
            "bet_actions": action_types.count("bet") + action_types.count("raise"),
            "call_actions": action_types.count("call"),
            "fold_actions": action_types.count("fold"),
            "avg_bet_size": sum(bet_sizes) / len(bet_sizes) if bet_sizes else 0,
            "min_bet_size": min(bet_sizes) if bet_sizes else 0,
            "max_bet_size": max(bet_sizes) if bet_sizes else 0,
        }

        return analysis

    def _estimate_hand_range(
        self, stats: Dict[str, Any], betting_analysis: Dict[str, Any]
    ) -> str:
        """Estimate opponent's hand range based on stats and betting."""
        vpip = stats.get("vpip_pct", 0)
        aggression = stats.get("aggression_factor", 0)

        # Simple range estimation based on VPIP
        if vpip < 15:
            return "very_tight"  # Top 15% of hands
        elif vpip < 25:
            return "tight"  # Top 25% of hands
        elif vpip < 40:
            return "medium"  # Top 40% of hands
        elif vpip < 60:
            return "loose"  # Top 60% of hands
        else:
            return "very_loose"  # Most hands

        # This could be enhanced with more sophisticated range estimation


def test_hand_tracker() -> None:
    """Test the hand tracker functionality."""
    print("🎯 Testing Hand Tracker...")
    print("=" * 40)

    tracker = HandTracker()

    # Start a new hand
    table_info = {"table_name": "Test Table", "stakes": "1/2"}
    tracker.start_new_hand("hand_001", table_info)

    # Add players
    tracker.add_player_to_hand("Player1", "UTG", 100.0)
    tracker.add_player_to_hand("Player2", "CO", 150.0)
    tracker.add_player_to_hand("Hero", "BB", 200.0)

    # Record some actions
    tracker.record_action("Player1", "raise", 6.0, "preflop")
    tracker.record_action("Player2", "call", 6.0, "preflop")
    tracker.record_action("Hero", "call", 4.0, "preflop")

    # Get opponent stats
    stats = tracker.get_opponent_stats("Player1")
    print(
        f"Player1 stats: VPIP {stats.get('vpip_pct', 0):.1f}%, "
        f"PFR {stats.get('pfr_pct', 0):.1f}%"
    )

    # Get hand context
    context = tracker.get_hand_context_for_ai()
    print(
        f"Hand context: {len(context['opponents'])} opponents, "
        f"{len(context['action_history'])} actions"
    )

    # End hand
    tracker.end_hand()

    tracker.close()
    print("✅ Hand tracker test completed!")


if __name__ == "__main__":
    test_hand_tracker()
