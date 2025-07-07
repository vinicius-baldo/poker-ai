"""
Core poker engine for hand evaluation and game state management.
"""
import logging
from enum import Enum
from typing import Any, Dict, List, Tuple

from treys import Card as TreysCard
from treys import Evaluator

logger = logging.getLogger(__name__)


class Suit(Enum):
    """Card suits."""

    HEARTS = "h"
    DIAMONDS = "d"
    CLUBS = "c"
    SPADES = "s"


class Rank(Enum):
    """Card ranks."""

    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14


class Street(Enum):
    """Poker streets."""

    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"


class Action(Enum):
    """Player actions."""

    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "all_in"


class Card:
    """Represents a playing card."""

    def __init__(self, rank: Rank, suit: Suit) -> None:
        self.rank = rank
        self.suit = suit

    def __str__(self) -> str:
        rank_str = str(self.rank.value) if self.rank.value <= 10 else self.rank.name[0]
        return f"{rank_str}{self.suit.value}"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return False
        return self.rank == other.rank and self.suit == other.suit

    def __hash__(self) -> int:
        return hash((self.rank.value, self.suit.value))


class Hand:
    """Represents a poker hand."""

    def __init__(self, cards: List[Card]) -> None:
        if len(cards) != 2:
            raise ValueError("Poker hand must contain exactly 2 cards")
        self.cards = sorted(cards, key=lambda c: c.rank.value, reverse=True)

    def __str__(self) -> str:
        return f"{self.cards[0]}{self.cards[1]}"

    def is_pair(self) -> bool:
        """Check if hand is a pair."""
        return self.cards[0].rank == self.cards[1].rank

    def is_suited(self) -> bool:
        """Check if hand is suited."""
        return self.cards[0].suit == self.cards[1].suit

    def is_connected(self) -> bool:
        """Check if hand is connected (adjacent ranks)."""
        return abs(self.cards[0].rank.value - self.cards[1].rank.value) == 1

    def is_broadway(self) -> bool:
        """Check if both cards are broadway (T, J, Q, K, A)."""
        return all(card.rank.value >= 10 for card in self.cards)


class GameState:
    """Tracks the current state of a poker game."""

    def __init__(self) -> None:
        self.street = Street.PREFLOP
        self.community_cards: List[Card] = []
        self.pot_size = 0
        self.current_bet = 0
        self.players: Dict[str, Dict] = {}
        self.action_history: List[Dict] = []
        self.button_position = 0
        self.current_player = ""

    def add_community_card(self, card: Card) -> None:
        """Add a community card."""
        if len(self.community_cards) >= 5:
            raise ValueError("Cannot add more than 5 community cards")
        self.community_cards.append(card)
        self._update_street()

    def _update_street(self) -> None:
        """Update the current street based on community cards."""
        card_count = len(self.community_cards)
        if card_count == 0:
            self.street = Street.PREFLOP
        elif card_count == 3:
            self.street = Street.FLOP
        elif card_count == 4:
            self.street = Street.TURN
        elif card_count == 5:
            self.street = Street.RIVER

    def add_player(self, name: str, stack: int, position: int) -> None:
        """Add a player to the game."""
        self.players[name] = {
            "stack": stack,
            "position": position,
            "folded": False,
            "all_in": False,
            "current_bet": 0,
        }

    def record_action(self, player: str, action: Action, amount: int = 0) -> None:
        """Record a player action."""
        action_record = {
            "player": player,
            "action": action,
            "amount": amount,
            "street": self.street,
            "timestamp": None,  # TODO: Add timestamp
        }
        self.action_history.append(action_record)

        # Update player state
        if player in self.players:
            if action == Action.FOLD:
                self.players[player]["folded"] = True
            elif action == Action.ALL_IN:
                self.players[player]["all_in"] = True
                self.players[player]["current_bet"] = self.players[player]["stack"]
            elif action in [Action.BET, Action.RAISE, Action.CALL]:
                self.players[player]["current_bet"] = amount

    def get_active_players(self) -> List[str]:
        """Get list of players who haven't folded."""
        return [name for name, data in self.players.items() if not data["folded"]]

    def get_pot_odds(self, call_amount: int) -> float:
        """Calculate pot odds for a call."""
        if call_amount == 0:
            return float("inf")
        return self.pot_size / call_amount


class HandEvaluator:
    """Evaluates poker hand strength."""

    evaluator = Evaluator()

    @staticmethod
    def _to_treys(card: Card) -> Any:
        """Convert our Card to treys Card integer."""
        rank_map = {
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "T",
            11: "J",
            12: "Q",
            13: "K",
            14: "A",
        }
        suit_map = {"h": "h", "d": "d", "c": "c", "s": "s"}
        rank = rank_map[card.rank.value]
        suit = suit_map[card.suit.value]
        result = TreysCard.new(f"{rank}{suit}")
        return result

    @classmethod
    def evaluate_hand(
        cls, hole_cards: List[Card], community_cards: List[Card]
    ) -> Tuple[int, str]:
        """
        Evaluate hand strength using treys.
        Returns (hand_rank, hand_name) where lower hand_rank is better.
        """
        if len(hole_cards) != 2 or len(community_cards) < 3:
            return (0, "Incomplete hand")
        treys_hole = [cls._to_treys(c) for c in hole_cards]
        treys_board = [cls._to_treys(c) for c in community_cards]
        score = cls.evaluator.evaluate(treys_board, treys_hole)
        class_string = cls.evaluator.class_to_string(
            cls.evaluator.get_rank_class(score)
        )
        return (score, class_string)

    @classmethod
    def calculate_equity(
        cls,
        hole_cards: List[Card],
        community_cards: List[Card],
        opponent_range: List[str],
    ) -> float:
        """
        Calculate equity against opponent range using treys (simple Monte Carlo).
        Returns equity as percentage (0-100).
        """
        # TODO: Implement full Monte Carlo simulation for equity
        # For now, return a placeholder
        return 50.0  # Placeholder
