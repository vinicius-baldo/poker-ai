#!/usr/bin/env python3
"""
Example usage of the LLM-based Poker Table Analyzer
"""

from src.main_llm_analyzer import LLMPokerAnalyzer


def main():
    # Initialize the analyzer
    analyzer = LLMPokerAnalyzer()

    # Setup your LLM API (choose one):

    # Option 1: OpenAI GPT-4V
    # analyzer.setup_llm_api("your-openai-api-key", "openai")

    # Option 2: Anthropic Claude
    # analyzer.setup_llm_api("your-anthropic-api-key", "anthropic")

    # Run continuous analysis (every 1 second)
    # analyzer.run_continuous_analysis(interval_seconds=1)

    # Or run just one analysis cycle
    # success = analyzer.run_analysis_cycle()
    # print(f"Analysis successful: {success}")

    print("LLM Poker Analyzer Example")
    print("=" * 40)
    print("To use this analyzer:")
    print("1. Get an API key from OpenAI or Anthropic")
    print("2. Uncomment the setup_llm_api() line above")
    print("3. Uncomment the run_continuous_analysis() line")
    print("4. Run this script")
    print("\nThe analyzer will:")
    print("- Take screenshots every second")
    print("- Send them to the LLM for analysis")
    print("- Extract player names, stacks, cards, actions")
    print("- Save everything to the local database")
    print("- Use existing player profiles to improve analysis")


if __name__ == "__main__":
    main()
