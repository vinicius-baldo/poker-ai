#!/usr/bin/env python3
"""
Test script to verify PokerAI setup and basic functionality.
"""
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_imports() -> bool:
    """Test that all modules can be imported."""
    try:
        # Test core imports
        from core.poker_engine import Card, GameState, Hand, Rank, Suit  # noqa: F401

        print("✓ Core poker engine imports successful")

        # Test vision imports
        from vision.screen_capture import PokerStarsRegions, ScreenCapture  # noqa: F401

        print("✓ Screen capture imports successful")

        # Test card recognizer imports
        from vision.card_recognizer import CardRecognizer, TemplateManager  # noqa: F401

        print("✓ Card recognizer imports successful")

        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_basic_functionality() -> bool:
    """Test basic functionality."""
    try:
        from core.poker_engine import Card, GameState, Hand, Rank, Suit

        # Test card creation
        ace_hearts = Card(Rank.ACE, Suit.HEARTS)
        king_spades = Card(Rank.KING, Suit.SPADES)
        print(f"✓ Created cards: {ace_hearts}, {king_spades}")

        # Test hand creation
        hand = Hand([ace_hearts, king_spades])
        print(f"✓ Created hand: {hand}")
        print(f"  - Is suited: {hand.is_suited()}")
        print(f"  - Is broadway: {hand.is_broadway()}")

        # Test game state
        game = GameState()
        game.add_player("Player1", 1000, 0)
        game.add_player("Player2", 1500, 1)
        print(f"✓ Created game with {len(game.players)} players")

        return True
    except Exception as e:
        print(f"✗ Functionality test error: {e}")
        return False


def test_screen_capture() -> bool:
    """Test screen capture setup."""
    try:
        from vision.screen_capture import PokerStarsRegions, ScreenCapture

        capture = ScreenCapture()
        PokerStarsRegions.setup_regions(capture)
        print(f"✓ Screen capture setup successful with {len(capture.regions)} regions")

        return True
    except Exception as e:
        print(f"✗ Screen capture test error: {e}")
        return False


def main() -> None:
    """Run all tests."""
    print("Testing PokerAI setup...\n")

    tests = [
        ("Import Test", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Screen Capture Setup", test_screen_capture),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        if test_func():
            passed += 1
        print()

    print(f"Tests completed: {passed}/{total} passed")

    if passed == total:
        print("🎉 All tests passed! Setup is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")


if __name__ == "__main__":
    main()
