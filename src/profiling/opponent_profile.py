"""
OpponentProfile: Tracks lifetime stats and hand history for a single opponent.
"""
import json
from typing import Dict, List, Optional


class OpponentProfile:
    """Tracks lifetime stats and hand history for a single opponent."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.hands_played = 0
        self.vpip = 0  # Voluntarily Put $ In Pot
        self.pfr = 0  # Preflop Raise
        self.three_bet = 0
        self.aggressive_actions = 0  # Bets + Raises
        self.calls = 0
        self.showdowns = 0
        self.went_to_showdown = 0
        self.hands_at_showdown: List[
            List[str]
        ] = []  # List of hands shown at showdown (e.g., ['Ah', 'Kd'])
        self.action_history: List[List[Dict]] = []  # List of actions per hand

    def record_hand(
        self,
        actions: List[Dict],
        went_to_showdown: bool = False,
        showdown_hand: Optional[List[str]] = None,
    ) -> None:
        """Record a hand and update stats based on actions."""
        self.hands_played += 1
        voluntarily_played = False
        preflop_raised = False
        three_bet = False
        aggressive = 0
        calls = 0
        for action in actions:
            street = action.get("street")
            act = action.get("action")
            if street == "preflop":
                if act in ("call", "bet", "raise"):  # VPIP
                    voluntarily_played = True
                if act == "raise":
                    if preflop_raised:
                        three_bet = True
                    preflop_raised = True
            if act in ("bet", "raise"):
                aggressive += 1
            if act == "call":
                calls += 1
        if voluntarily_played:
            self.vpip += 1
        if preflop_raised:
            self.pfr += 1
        if three_bet:
            self.three_bet += 1
        self.aggressive_actions += aggressive
        self.calls += calls
        if went_to_showdown:
            self.went_to_showdown += 1
            if showdown_hand:
                self.showdowns += 1
                self.hands_at_showdown.append(showdown_hand)
        self.action_history.append(actions)

    def get_stats(self) -> Dict:
        """Return current stats as a dictionary."""
        stats = {
            "name": self.name,
            "hands_played": self.hands_played,
            "vpip_pct": (self.vpip / self.hands_played * 100)
            if self.hands_played
            else 0,
            "pfr_pct": (self.pfr / self.hands_played * 100) if self.hands_played else 0,
            "three_bet_pct": (self.three_bet / self.hands_played * 100)
            if self.hands_played
            else 0,
            "aggression_factor": (self.aggressive_actions / self.calls)
            if self.calls
            else 0,
            "showdown_pct": (self.showdowns / self.hands_played * 100)
            if self.hands_played
            else 0,
            "wtsd_pct": (self.went_to_showdown / self.hands_played * 100)
            if self.hands_played
            else 0,
            "hands_at_showdown": self.hands_at_showdown,
        }
        return stats

    def to_json(self) -> str:
        return json.dumps(self.get_stats())

    @staticmethod
    def from_json(data: str) -> "OpponentProfile":
        stats = json.loads(data)
        profile = OpponentProfile(stats["name"])
        profile.hands_played = stats["hands_played"]
        profile.vpip = int(stats.get("vpip_pct", 0) * profile.hands_played / 100)
        profile.pfr = int(stats.get("pfr_pct", 0) * profile.hands_played / 100)
        profile.three_bet = int(
            stats.get("three_bet_pct", 0) * profile.hands_played / 100
        )
        profile.aggressive_actions = int(
            stats.get("aggression_factor", 0) * profile.calls
        )
        profile.calls = profile.calls or 1  # Avoid division by zero
        profile.showdowns = int(
            stats.get("showdown_pct", 0) * profile.hands_played / 100
        )
        profile.went_to_showdown = int(
            stats.get("wtsd_pct", 0) * profile.hands_played / 100
        )
        profile.hands_at_showdown = stats.get("hands_at_showdown", [])
        return profile
