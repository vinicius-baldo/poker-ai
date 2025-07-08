#!/usr/bin/env python3
"""
Test script for turn detection and LLM analysis
"""

import time

from src.main_llm_analyzer import MainLLMAnalyzer
from src.vision.table_analyzer import TableAnalyzer


def test_turn_detection():
    """Test turn detection with a sample image"""
    print("🎯 Testing Turn Detection")
    print("=" * 40)

    analyzer = TableAnalyzer()

    # Test with your screenshot
    test_image = "imagem_teste.png"

    try:
        import cv2

        image = cv2.imread(test_image)
        if image is not None:
            is_turn = analyzer.detect_player_turn(image)
            print(f"Turn detected: {is_turn}")
            print(f"Confidence: {analyzer.turn_detection_confidence:.2f}")
            print(f"Active buttons: {analyzer.last_action_buttons}")
        else:
            print(f"Could not load image: {test_image}")
    except Exception as e:
        print(f"Error testing turn detection: {e}")


def test_llm_analysis():
    """Test the complete LLM analysis system"""
    print("\n🤖 Testing LLM Analysis System")
    print("=" * 40)

    analyzer = MainLLMAnalyzer()

    # Run one analysis cycle
    print("Running analysis cycle...")
    success = analyzer.run_analysis_cycle()

    if success:
        print("✅ Analysis completed successfully")
    else:
        print("❌ Analysis failed or skipped (not player's turn)")


def test_continuous_analysis():
    """Test continuous analysis for a few cycles"""
    print("\n🔄 Testing Continuous Analysis")
    print("=" * 40)
    print("This will run for 10 seconds, analyzing only when it's your turn...")
    print("Press Ctrl+C to stop early")

    analyzer = MainLLMAnalyzer()

    start_time = time.time()
    cycle_count = 0

    try:
        while time.time() - start_time < 10:  # Run for 10 seconds
            cycle_count += 1
            print(f"\n--- Cycle {cycle_count} ---")

            success = analyzer.run_analysis_cycle()
            if success:
                print("✅ Analysis completed")
            else:
                print("⏭️  Skipped (not player's turn)")

            time.sleep(1)  # Wait 1 second between cycles

    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")

    print(f"\n📊 Summary:")
    print(f"  Total cycles: {cycle_count}")
    print(f"  Duration: {time.time() - start_time:.1f} seconds")


def main():
    """Main test function"""
    print("🧪 PokerAI Turn Detection & LLM Analysis Test")
    print("=" * 50)

    # Test turn detection
    test_turn_detection()

    # Test LLM analysis (requires API setup)
    print("\n" + "=" * 50)
    print("Note: LLM analysis requires API keys to be configured")
    print("Set up your LLM API in the analyzer classes before testing")

    # Uncomment these to test (after setting up API keys):
    # test_llm_analysis()
    # test_continuous_analysis()

    print("\n✅ Test completed!")


if __name__ == "__main__":
    main()
