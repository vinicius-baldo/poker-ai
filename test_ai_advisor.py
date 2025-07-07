#!/usr/bin/env python3
# flake8: noqa: E402
"""
Test script for AI Poker Advisor.
"""
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ai.poker_advisor import PokerAdvisor, PokerAnalysisResult
from core.poker_engine import Card, Rank, Suit


def test_ai_advisor() -> None:
    """Test the AI advisor with sample poker situations."""

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return

    print("🤖 Testing AI Poker Advisor...")
    print("=" * 50)

    # Initialize advisor
    advisor = PokerAdvisor(api_key=api_key, model="gpt-4")

    # Test situation 1: Strong hand preflop
    print("\n📊 Test 1: Strong hand preflop")
    print("-" * 30)

    hole_cards = [Card(Rank.ACE, Suit.HEARTS), Card(Rank.ACE, Suit.DIAMONDS)]
    community_cards = []

    analysis = advisor.analyze_situation(
        hole_cards=hole_cards,
        community_cards=community_cards,
        pot_size=30.0,
        current_bet=10.0,
        player_stack=100.0,
        opponent_stack=80.0,
        position="button",
        street="preflop",
        action_history=[
            {"player": "UTG", "action": "raise", "amount": 10},
            {"player": "MP", "action": "call", "amount": 10},
        ],
    )

    result = PokerAnalysisResult(analysis)
    print(f"🎯 Recommendation: {result.recommendation.upper()}")
    print(f"📈 Confidence: {result.confidence:.1%}")
    print(f"💭 Reasoning: {result.reasoning}")
    print(f"💰 Expected Value: ${result.expected_value:.2f}")
    print(f"⚠️  Risk Level: {result.risk_level}")
    print(f"🃏 Hand Strength: {result.hand_strength}")

    # Test situation 2: Flop with medium hand
    print("\n📊 Test 2: Medium hand on flop")
    print("-" * 30)

    hole_cards = [Card(Rank.KING, Suit.HEARTS), Card(Rank.QUEEN, Suit.HEARTS)]
    community_cards = [
        Card(Rank.ACE, Suit.SPADES),
        Card(Rank.TEN, Suit.HEARTS),
        Card(Rank.TWO, Suit.CLUBS),
    ]

    analysis = advisor.analyze_situation(
        hole_cards=hole_cards,
        community_cards=community_cards,
        pot_size=60.0,
        current_bet=20.0,
        player_stack=150.0,
        opponent_stack=120.0,
        position="cutoff",
        street="flop",
        action_history=[
            {"player": "UTG", "action": "raise", "amount": 15},
            {"player": "MP", "action": "call", "amount": 15},
            {"player": "Hero", "action": "call", "amount": 15},
            {"player": "UTG", "action": "bet", "amount": 20},
        ],
    )

    result = PokerAnalysisResult(analysis)
    print(f"🎯 Recommendation: {result.recommendation.upper()}")
    print(f"📈 Confidence: {result.confidence:.1%}")
    print(f"💭 Reasoning: {result.reasoning}")
    print(f"💰 Expected Value: ${result.expected_value:.2f}")
    print(f"⚠️  Risk Level: {result.risk_level}")
    print(f"🃏 Hand Strength: {result.hand_strength}")
    print(f"📊 Pot Odds: {result.pot_odds:.2f}:1")

    # Test situation 3: Weak hand, facing aggression
    print("\n📊 Test 3: Weak hand, facing aggression")
    print("-" * 30)

    hole_cards = [Card(Rank.SEVEN, Suit.CLUBS), Card(Rank.TWO, Suit.DIAMONDS)]
    community_cards = [
        Card(Rank.ACE, Suit.SPADES),
        Card(Rank.KING, Suit.HEARTS),
        Card(Rank.QUEEN, Suit.DIAMONDS),
        Card(Rank.JACK, Suit.CLUBS),
    ]

    analysis = advisor.analyze_situation(
        hole_cards=hole_cards,
        community_cards=community_cards,
        pot_size=100.0,
        current_bet=50.0,
        player_stack=200.0,
        opponent_stack=300.0,
        position="big_blind",
        street="turn",
        action_history=[
            {"player": "UTG", "action": "raise", "amount": 20},
            {"player": "Hero", "action": "call", "amount": 20},
            {"player": "UTG", "action": "bet", "amount": 30},
            {"player": "Hero", "action": "call", "amount": 30},
            {"player": "UTG", "action": "bet", "amount": 50},
        ],
    )

    result = PokerAnalysisResult(analysis)
    print(f"🎯 Recommendation: {result.recommendation.upper()}")
    print(f"📈 Confidence: {result.confidence:.1%}")
    print(f"💭 Reasoning: {result.reasoning}")
    print(f"💰 Expected Value: ${result.expected_value:.2f}")
    print(f"⚠️  Risk Level: {result.risk_level}")
    print(f"🃏 Hand Strength: {result.hand_strength}")

    # Show GUI format
    print("\n🖥️  GUI Format Example:")
    print("-" * 30)
    gui_data = result.to_gui_format()
    print("Primary Action:")
    print(f"  Action: {gui_data['primary_action']['action']}")
    print(f"  Color: {result.get_action_color()}")
    print(f"  Confidence Color: {result.get_confidence_color()}")
    print(f"  Reasoning: {gui_data['primary_action']['reasoning']}")

    print("\nHand Info:")
    print(f"  Strength: {gui_data['hand_info']['strength']}")
    print(f"  Pot Odds: {gui_data['hand_info']['pot_odds']:.2f}:1")
    print(f"  Stack/Pot Ratio: {gui_data['hand_info']['stack_to_pot_ratio']:.2f}")

    print("\nAlternative Actions:")
    for alt in gui_data["alternatives"]:
        print(f"  - {alt['action'].upper()}: {alt['confidence']:.1%} confidence")

    print("\n✅ AI Advisor test completed!")


def test_without_api() -> None:
    """Test the system without API key (fallback mode)."""
    print("🧪 Testing AI Advisor in fallback mode (no API key)...")
    print("=" * 50)

    # Create advisor with dummy key (will use fallback)
    advisor = PokerAdvisor(api_key="dummy-key", model="gpt-4")

    hole_cards = [Card(Rank.ACE, Suit.HEARTS), Card(Rank.KING, Suit.HEARTS)]
    community_cards = [
        Card(Rank.QUEEN, Suit.HEARTS),
        Card(Rank.JACK, Suit.HEARTS),
        Card(Rank.TEN, Suit.HEARTS),
    ]

    analysis = advisor.analyze_situation(
        hole_cards=hole_cards,
        community_cards=community_cards,
        pot_size=50.0,
        current_bet=25.0,
        player_stack=100.0,
        opponent_stack=75.0,
        position="button",
        street="flop",
    )

    result = PokerAnalysisResult(analysis)
    print(f"🎯 Recommendation: {result.recommendation.upper()}")
    print(f"📈 Confidence: {result.confidence:.1%}")
    print(f"💭 Reasoning: {result.reasoning}")
    print(f"🃏 Hand Strength: {result.hand_strength}")
    print(f"📊 Pot Odds: {result.pot_odds:.2f}:1")

    print("\n✅ Fallback test completed!")


def main() -> None:
    """Main function."""
    if len(sys.argv) > 1 and sys.argv[1] == "--fallback":
        test_without_api()
    else:
        test_ai_advisor()


if __name__ == "__main__":
    main()
