#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

# flake8: noqa: E402
"""
Test script for high priority features:
1. Screen capture
2. Position detection
3. Hand tracking integration
"""
import time
from typing import Optional

from analysis.hand_tracker import HandTracker
from main_poker_assistant import PokerAssistant
from vision.position_detector import PositionDetector
from vision.screen_capture import PokerTableCapture, ScreenCapture


def test_screen_capture() -> None:
    """Test screen capture functionality."""
    print("🖥️ Testing Screen Capture...")
    print("=" * 50)

    # Test basic screen capture
    capture = ScreenCapture()

    # List monitors
    monitors = capture.list_monitors()
    print(f"📺 Found {len(monitors)} monitor(s)")

    # Test capture
    print("📸 Capturing screen...")
    img = capture.capture_screen()
    if img is not None:
        print(f"✅ Captured image: {img.shape}")

        # Save test image
        import cv2

        cv2.imwrite("test_screen_capture.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print("💾 Saved as 'test_screen_capture.png'")
    else:
        print("❌ Failed to capture screen")

    capture.close()
    print("✅ Screen capture test completed!\n")


def test_position_detector() -> None:
    """Test position detection functionality."""
    print("🎯 Testing Position Detector...")
    print("=" * 50)

    detector = PositionDetector("9max")

    # Test position advantages
    positions = ["BTN", "CO", "MP", "UTG", "BB"]
    print("📊 Position Advantages:")
    for pos in positions:
        advantage = detector.get_position_advantage(pos)
        print(f"  {pos}: {advantage}")

    # Test relative positions
    print("\n🔄 Relative Positions:")
    hero_pos = "CO"
    opponent_pos = "UTG"
    relative = detector.get_relative_position(hero_pos, opponent_pos)
    print(f"  Hero ({hero_pos}) vs Opponent ({opponent_pos}): {relative}")

    # Test position stats
    stats = detector.get_position_stats()
    print(f"\n📈 Position Stats: {stats}")

    print("✅ Position detector test completed!\n")


def test_hand_tracker() -> None:
    """Test hand tracking and profiling integration."""
    print("🎯 Testing Hand Tracker...")
    print("=" * 50)

    tracker = HandTracker()

    # Start a new hand
    table_info = {"table_name": "Test Table", "stakes": "1/2"}
    tracker.start_new_hand("test_hand_001", table_info)

    # Add players
    tracker.add_player_to_hand("Player1", "UTG", 100.0)
    tracker.add_player_to_hand("Player2", "CO", 150.0)
    tracker.add_player_to_hand("Player3", "BTN", 200.0)
    tracker.add_player_to_hand("Hero", "BB", 250.0)

    # Record actions
    print("📝 Recording actions...")
    tracker.record_action("Player1", "raise", 6.0, "preflop")
    tracker.record_action("Player2", "call", 6.0, "preflop")
    tracker.record_action("Player3", "fold", 0.0, "preflop")
    tracker.record_action("Hero", "call", 4.0, "preflop")

    # Get opponent stats
    print("\n👥 Opponent Analysis:")
    for player in ["Player1", "Player2", "Player3"]:
        stats = tracker.get_opponent_stats(player)
        print(
            f"  {player}: VPIP {stats.get('vpip_pct', 0):.1f}%, "
            f"PFR {stats.get('pfr_pct', 0):.1f}%, "
            f"Style: {stats.get('playing_style', 'unknown')}"
        )

    # Get hand context
    context = tracker.get_hand_context_for_ai()
    print(
        f"\n🎮 Hand Context: {len(context['opponents'])} opponents, "
        f"{len(context['action_history'])} actions"
    )

    # End hand
    tracker.end_hand()

    # Get session stats
    session_stats = tracker.get_session_stats()
    print(f"\n📊 Session Stats: {session_stats}")

    tracker.close()
    print("✅ Hand tracker test completed!\n")


def test_integrated_system() -> None:
    """Test the integrated poker assistant with new features."""
    print("🤖 Testing Integrated Poker Assistant...")
    print("=" * 50)

    assistant = PokerAssistant()

    # Test with existing image if available
    if os.path.exists("imagem_tela.png"):
        print("📸 Testing with existing image...")

        # Start new hand
        assistant.start_new_hand()

        # Analyze situation
        result = assistant.analyze_current_situation("imagem_tela.png")

        if result:
            print(f"🎯 AI Recommendation: {result.recommendation.upper()}")
            print(f"📈 Confidence: {result.confidence:.1%}")
            print(f"💭 Reasoning: {result.reasoning[:100]}...")

        # Get GUI data
        gui_data = assistant.get_gui_data("imagem_tela.png")
        print(f"🖥️ GUI Data keys: {list(gui_data.keys())}")

        # End hand
        assistant.end_current_hand()

        # Get session stats
        session_stats = assistant.get_session_stats()
        print(f"📊 Session Stats: {session_stats}")
    else:
        print("⚠️ No test image found, skipping image analysis")

    # Test calibration (interactive)
    print("\n🎯 Testing Table Region Calibration...")
    print("This will open a window to select table region (press ESC to skip)")
    try:
        region = assistant.calibrate_table_region()
        if region:
            print(f"✅ Calibrated region: {region}")
        else:
            print("⚠️ Calibration cancelled or failed")
    except Exception as e:
        print(f"❌ Calibration error: {e}")

    assistant.close()
    print("✅ Integrated system test completed!\n")


def test_real_time_features() -> None:
    """Test real-time monitoring features."""
    print("⏱️ Testing Real-time Features...")
    print("=" * 50)

    assistant = PokerAssistant()

    print("🔄 Testing real-time monitoring (5 seconds)...")
    print("This will capture and analyze screens every 2 seconds")

    try:
        # Start monitoring
        assistant.start_real_time_monitoring()

        # Let it run for 5 seconds
        time.sleep(5)

        # Stop monitoring
        assistant.stop_real_time_monitoring()

        print("✅ Real-time monitoring test completed")

    except Exception as e:
        print(f"❌ Real-time monitoring error: {e}")

    assistant.close()
    print("✅ Real-time features test completed!\n")


def main() -> None:
    """Run all high priority feature tests."""
    print("🚀 Testing High Priority Features")
    print("=" * 60)
    print()

    # Test individual components
    test_screen_capture()
    test_position_detector()
    test_hand_tracker()

    # Test integrated system
    test_integrated_system()

    # Test real-time features (optional)
    response = input("Test real-time monitoring? (y/n): ").lower().strip()
    if response == "y":
        test_real_time_features()

    print("🎉 All high priority feature tests completed!")
    print("\n📋 Summary of implemented features:")
    print("✅ Screen capture with rate limiting")
    print("✅ Position detection and analysis")
    print("✅ Hand tracking with opponent profiling")
    print("✅ Real-time monitoring capability")
    print("✅ Integrated system with all components")


if __name__ == "__main__":
    main()
