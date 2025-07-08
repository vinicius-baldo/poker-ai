#!/usr/bin/env python3
"""
Main LLM-based Poker Table Analyzer
Takes screenshots every second and uses LLM vision to extract player information
"""

import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from vision.screen_capture import ScreenCapture
from vision.table_analyzer import TableAnalyzer


class LLMPokerAnalyzer:
    def __init__(self, config_path: str = "config/table_regions.json"):
        self.config_path = config_path
        self.table_analyzer = TableAnalyzer(config_path)
        self.screen_capture = ScreenCapture()

        # Initialize database
        self.init_database()

    def init_database(self):
        """Initialize the database with the required schema"""
        db_path = "data/opponents.db"
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Read and execute schema
        schema_path = "src/profiling/database_schema.sql"
        if os.path.exists(schema_path):
            with open(schema_path, "r") as f:
                schema = f.read()

            conn = sqlite3.connect(db_path)
            conn.executescript(schema)
            conn.close()
            print("Database initialized successfully")
        else:
            print(f"Warning: Schema file not found at {schema_path}")

    def capture_screenshot(self) -> bytes:
        """Capture screenshot using the existing screen capture module"""
        try:
            # Use the existing screen capture functionality
            screenshot = self.screen_capture.capture_table_region()
            if screenshot is not None:
                # Convert to bytes for LLM API
                import cv2

                success, buffer = cv2.imencode(".jpg", screenshot)
                if success:
                    return buffer.tobytes()
        except Exception as e:
            print(f"Error capturing screenshot: {e}")
        return None

    def setup_llm_api(self, api_key: str, provider: str = "openai"):
        """Setup LLM API credentials"""
        self.table_analyzer.llm_api_key = api_key
        self.table_analyzer.llm_provider = provider

        if provider == "openai":
            self.table_analyzer.llm_endpoint = (
                "https://api.openai.com/v1/chat/completions"
            )
        elif provider == "anthropic":
            self.table_analyzer.llm_endpoint = "https://api.anthropic.com/v1/messages"
        # Add other providers as needed

    def implement_llm_call(self, image_base64: str, prompt: str):
        """Implement the actual LLM API call based on provider"""
        if not self.table_analyzer.llm_api_key:
            print("LLM API key not set. Please call setup_llm_api() first.")
            return None

        try:
            import requests

            if self.table_analyzer.llm_provider == "openai":
                return self._call_openai_api(image_base64, prompt)
            elif self.table_analyzer.llm_provider == "anthropic":
                return self._call_anthropic_api(image_base64, prompt)
            else:
                print(f"Unsupported LLM provider: {self.table_analyzer.llm_provider}")
                return None

        except Exception as e:
            print(f"Error calling LLM API: {e}")
            return None

    def _call_openai_api(self, image_base64: str, prompt: str):
        """Call OpenAI GPT-4V API"""
        headers = {
            "Authorization": f"Bearer {self.table_analyzer.llm_api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 1000,
        }

        response = requests.post(
            self.table_analyzer.llm_endpoint, headers=headers, json=payload, timeout=30
        )
        return response.json()

    def _call_anthropic_api(self, image_base64: str, prompt: str):
        """Call Anthropic Claude API"""
        headers = {
            "x-api-key": self.table_analyzer.llm_api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        payload = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64,
                            },
                        },
                    ],
                }
            ],
        }

        response = requests.post(
            self.table_analyzer.llm_endpoint, headers=headers, json=payload, timeout=30
        )
        return response.json()

    def parse_llm_response(self, llm_response: dict) -> dict:
        """Parse LLM response to extract JSON analysis"""
        try:
            if self.table_analyzer.llm_provider == "openai":
                content = llm_response["choices"][0]["message"]["content"]
            elif self.table_analyzer.llm_provider == "anthropic":
                content = llm_response["content"][0]["text"]
            else:
                return None

            # Extract JSON from the response
            import json
            import re

            # Try to find JSON in the response
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                print("No JSON found in LLM response")
                return None

        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return None

    def run_analysis_cycle(self) -> bool:
        """Run one complete analysis cycle"""
        try:
            # Capture screenshot
            screenshot = self.capture_screenshot()
            if not screenshot:
                print("Failed to capture screenshot")
                return False

            # Convert to OpenCV format for turn detection
            import cv2
            import numpy as np

            nparr = np.frombuffer(screenshot, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Check if it's the player's turn
            is_player_turn = self.table_analyzer.detect_player_turn(image)

            if not is_player_turn:
                print(
                    f"Not player's turn (confidence: {self.table_analyzer.turn_detection_confidence:.2f}), skipping analysis"
                )
                return False

            print(
                f"Player's turn detected! Available actions: {self.table_analyzer.last_action_buttons}"
            )

            # Get existing player profiles
            player_profiles = self.table_analyzer.get_player_profiles_from_db()

            # Create LLM prompt
            prompt = self.table_analyzer.create_llm_prompt(player_profiles)

            # Encode image
            image_base64 = self.table_analyzer.encode_image_for_llm(screenshot)

            # Call LLM
            llm_response = self.implement_llm_call(image_base64, prompt)
            if not llm_response:
                print("Failed to get LLM response")
                return False

            # Parse response
            analysis = self.parse_llm_response(llm_response)
            if not analysis:
                print("Failed to parse LLM response")
                return False

            # Add timestamp if not present
            if "timestamp" not in analysis:
                analysis["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Add turn detection info
            analysis["turn_detection"] = {
                "is_player_turn": is_player_turn,
                "confidence": self.table_analyzer.turn_detection_confidence,
                "detected_buttons": self.table_analyzer.last_action_buttons,
            }

            # Save to database
            self.table_analyzer.save_analysis_to_db(analysis)

            # Print summary
            print(f"Analysis completed at {analysis.get('timestamp')}")
            print(f"  Players detected: {len(analysis.get('players', []))}")
            print(
                f"  Pot size: {analysis.get('table_info', {}).get('pot_size', 'Unknown')}"
            )
            print(
                f"  Street: {analysis.get('table_info', {}).get('current_street', 'Unknown')}"
            )

            # Check if action is required
            action_required = analysis.get("action_required", {})
            if action_required.get("is_hero_turn"):
                available_actions = action_required.get("available_actions", [])
                current_bet = action_required.get("current_bet_to_call", "Unknown")
                print(f"  🎯 ACTION REQUIRED: {available_actions}")
                print(f"  💰 Bet to call: {current_bet}")

            return True

        except Exception as e:
            print(f"Error in analysis cycle: {e}")
            return False

    def run_continuous_analysis(self, interval_seconds: int = 1):
        """Run continuous table analysis"""
        print(
            f"Starting continuous LLM-based table analysis (every {interval_seconds} second(s))"
        )
        print("Press Ctrl+C to stop")

        cycle_count = 0
        success_count = 0

        while True:
            try:
                cycle_count += 1
                print(f"\n--- Cycle {cycle_count} ---")

                success = self.run_analysis_cycle()
                if success:
                    success_count += 1

                print(
                    f"Success rate: {success_count}/{cycle_count} ({success_count/cycle_count*100:.1f}%)"
                )

                time.sleep(interval_seconds)

            except KeyboardInterrupt:
                print("\nAnalysis stopped by user")
                break
            except Exception as e:
                print(f"Error in continuous analysis: {e}")
                time.sleep(interval_seconds)


def main():
    """Main function"""
    analyzer = LLMPokerAnalyzer()

    # Check if LLM API is configured
    if not analyzer.table_analyzer.llm_api_key:
        print("LLM API not configured. Please set up your API key:")
        print("1. Get an API key from OpenAI, Anthropic, or another provider")
        print("2. Call analyzer.setup_llm_api('your_api_key', 'provider')")
        print("3. Run analyzer.run_continuous_analysis()")

        # Example setup (uncomment and modify):
        # analyzer.setup_llm_api("your-api-key-here", "openai")
        # analyzer.run_continuous_analysis()

        return

    # Run analysis
    analyzer.run_continuous_analysis()


if __name__ == "__main__":
    main()
