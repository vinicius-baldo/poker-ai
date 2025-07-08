import base64
import json
import sqlite3
import time
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Optional

import cv2
import requests
from PIL import Image
import os
import subprocess


class TableAnalyzer:
    def __init__(
        self,
        config_path: str = "config/table_regions.json",
        db_path: str = "data/opponents.db",
    ):
        self.config_path = config_path
        self.db_path = db_path
        self.llm_api_key = None  # Set your LLM API key here
        self.llm_endpoint = None  # Set your LLM endpoint here

        # Load table configuration
        try:
            with open(config_path, "r") as f:
                self.config = json.load(f)
        except FileNotFoundError:
            print(f"Table config {config_path} not found, using defaults")
            self.config = {}

        # Load button regions configuration
        self.button_regions = self.load_button_regions()

        # Turn detection state
        self.last_action_buttons = None
        self.is_player_turn = False
        self.turn_detection_confidence = 0.0
        
        # State tracking to prevent duplicate analysis
        self.last_bet_amount = None
        self.last_analysis_timestamp = None
        self.last_analysis_hash = None

    def load_button_regions(self) -> Dict:
        """Load button regions from config file"""
        try:
            with open("config/button_regions.json", "r") as f:
                config = json.load(f)
            return config.get("button_regions", self.get_default_button_regions())
        except FileNotFoundError:
            print("Button regions config not found, using defaults")
            return self.get_default_button_regions()
        except Exception as e:
            print(f"Error loading button regions: {e}")
            return self.get_default_button_regions()

    def get_default_button_regions(self) -> Dict:
        """Get default button regions"""
        return {
            "fold": {"x": 300, "y": 550, "w": 80, "h": 30},
            "check_call": {"x": 400, "y": 550, "w": 80, "h": 30},
            "bet_raise": {"x": 500, "y": 550, "w": 80, "h": 30},
            "all_in": {"x": 600, "y": 550, "w": 80, "h": 30},
        }

    def detect_player_turn(self, image) -> bool:
        """
        Detect if it's the player's turn by looking for the bet slider.
        Returns True if it's the player's turn, False otherwise.
        """
        try:
            # Convert to grayscale for better detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Load slider region from config
            try:
                with open("config/slider_region.json", "r") as f:
                    config = json.load(f)
                slider_region = config.get("slider_region", {"x": 1385, "y": 1250, "w": 120, "h": 40})
            except FileNotFoundError:
                slider_region = {"x": 1385, "y": 1250, "w": 120, "h": 40}
            
            x, y, w, h = slider_region["x"], slider_region["y"], slider_region["w"], slider_region["h"]
            
            # Check if region is within image bounds
            if x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
                return False
            
            slider_area = gray[y:y+h, x:x+w]
            
            # Simple detection: look for bright/active slider area
            mean_brightness = cv2.mean(slider_area)[0]
            is_slider_present = mean_brightness > 80  # Threshold for "active" slider
            
            # Try to read bet amount from the region
            current_bet_amount = None
            if is_slider_present:
                try:
                    import pytesseract
                    # Preprocess the region for better OCR
                    slider_area_resized = cv2.resize(slider_area, (w*2, h*2))
                    _, thresh = cv2.threshold(slider_area_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    # OCR configuration for numbers
                    config = '--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.'
                    text = pytesseract.image_to_string(thresh, config=config).strip()
                    
                    # Extract number
                    import re
                    numbers = re.findall(r'\d+\.?\d*', text)
                    if numbers:
                        current_bet_amount = numbers[0]
                        
                except Exception as e:
                    print(f"Error reading bet amount: {e}")
            
            # Update state
            self.is_player_turn = is_slider_present
            self.turn_detection_confidence = 1.0 if is_slider_present else 0.0
            self.last_action_buttons = ["bet_slider"] if is_slider_present else []
            
            # Check if this is a new decision (bet amount changed)
            is_new_decision = False
            if is_slider_present and current_bet_amount != self.last_bet_amount:
                is_new_decision = True
                self.last_bet_amount = current_bet_amount
                print(f"New bet amount detected: {current_bet_amount}")
            
            return is_slider_present and is_new_decision
            
        except Exception as e:
            print(f"Error detecting player turn: {e}")
            return False

    def capture_screenshot(self) -> bytes:
        """Capture screenshot of the poker table"""
        # This would integrate with your existing screen capture
        # For now, placeholder - you'll need to implement this
        pass

    def encode_image_for_llm(self, image_bytes: bytes) -> str:
        """Encode image to base64 for LLM API"""
        return base64.b64encode(image_bytes).decode("utf-8")

    def get_player_profiles_from_db(self) -> Dict[str, Dict]:
        """Get existing player profiles from database"""
        profiles = {}
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT name, aggression_level, vpip, pfr, avg_stack,
                       total_hands, last_seen
                FROM player_profiles
            """
            )

            for row in cursor.fetchall():
                name, aggression, vpip, pfr, avg_stack, hands, last_seen = row
                profiles[name] = {
                    "aggression_level": aggression,
                    "vpip": vpip,
                    "pfr": pfr,
                    "avg_stack": avg_stack,
                    "total_hands": hands,
                    "last_seen": last_seen,
                }

            conn.close()
        except Exception as e:
            print(f"Error loading player profiles: {e}")

        return profiles

    def create_llm_prompt(self, player_profiles: Dict[str, Dict]) -> str:
        """Create prompt for LLM with context from player profiles"""
        prompt = """Analyze this poker table screenshot and extract the following information in JSON format:

REQUIRED OUTPUT FORMAT:
{
  "timestamp": "YYYY-MM-DD HH:MM:SS",
  "table_info": {
    "pot_size": "amount",
    "community_cards": ["card1", "card2", "card3", "card4", "card5"],
    "current_street": "preflop|flop|turn|river",
    "button_position": "seat_number"
  },
  "players": [
    {
      "seat": "seat_number",
      "name": "player_name",
      "stack": "stack_amount",
      "cards": ["card1", "card2"],
      "action": "fold|check|call|bet|raise|all_in",
      "bet_amount": "amount_if_betting",
      "is_active": true/false,
      "position": "UTG|UTG+1|MP|MP+1|CO|BTN|SB|BB"
    }
  ],
  "action_required": {
    "is_hero_turn": true/false,
    "available_actions": ["fold", "check", "call", "bet", "raise"],
    "current_bet_to_call": "amount",
    "min_raise": "amount",
    "max_raise": "amount"
  }
}

IMPORTANT RULES:
- Only include players who are visible/active at the table
- If cards are not visible, use null for cards array
- If player has folded, set is_active to false
- Use exact seat numbers (0-8)
- For bet amounts, include the actual number
- For stack amounts, include the actual number
- In action_required, specify what actions the hero can take

PLAYER CONTEXT (use this to improve analysis):
"""

        if player_profiles:
            prompt += "\nKNOWN PLAYERS:\n"
            for name, profile in player_profiles.items():
                prompt += f"- {name}: Aggression={profile['aggression_level']}, VPIP={profile['vpip']}%, PFR={profile['pfr']}%, Avg Stack={profile['avg_stack']}, Hands={profile['total_hands']}\n"
        else:
            prompt += "\nNo known players in database.\n"

        prompt += """
ANALYSIS TIPS:
- Look for player names in the top area of each seat
- Stack amounts are usually below the player name
- Cards are typically shown at the bottom of the seat area
- "All In" indicators are usually prominent
- Button position is marked with a "D" or similar indicator
- Community cards are in the center of the table
- Action buttons (fold, call, raise) indicate it's the hero's turn

Please provide only the JSON response, no additional text."""

        return prompt

    def play_llm_notification(self):
        """Play a sound notification when LLM is called (for debugging)"""
        try:
            # Try different sound methods based on OS
            if os.name == 'posix':  # macOS/Linux
                # Use afplay on macOS with a shorter timeout
                subprocess.run(['afplay', '/System/Library/Sounds/Ping.aiff'], 
                             capture_output=True, timeout=1)
            elif os.name == 'nt':  # Windows
                # Use winsound on Windows
                import winsound
                winsound.MessageBeep()
            else:
                # Fallback: print a bell character
                print('\a')
            print("🔔 LLM Analysis Triggered!")
        except subprocess.TimeoutExpired:
            # Sound played but timed out (normal)
            print("🔔 LLM Analysis Triggered!")
        except Exception as e:
            # Fallback: just print a notification
            print(f"🔔 LLM Analysis Triggered! (Sound failed: {e})")

    def call_llm_vision_api(self, image_base64: str, prompt: str) -> Optional[Dict]:
        """Call LLM vision API to analyze the image"""
        # This is a placeholder - you'll need to implement with your chosen LLM
        # Examples: OpenAI GPT-4V, Claude, Gemini, etc.

        # Example for OpenAI GPT-4V:
        """
        headers = {
            "Authorization": f"Bearer {self.llm_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                    ]
                }
            ],
            "max_tokens": 1000
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        return response.json()
        """

        # For now, return None - implement with your LLM choice
        return None

    def save_analysis_to_db(self, analysis: Dict):
        """Save the analysis results to local database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Save table state
            table_info = analysis.get("table_info", {})
            cursor.execute(
                """
                INSERT INTO table_states
                (timestamp, pot_size, community_cards, current_street, button_position)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    analysis.get("timestamp"),
                    table_info.get("pot_size"),
                    json.dumps(table_info.get("community_cards", [])),
                    table_info.get("current_street"),
                    table_info.get("button_position"),
                ),
            )

            table_state_id = cursor.lastrowid

            # Save player states
            for player in analysis.get("players", []):
                cursor.execute(
                    """
                    INSERT INTO player_states
                    (table_state_id, seat, name, stack, cards, action, bet_amount, is_active, position)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        table_state_id,
                        player.get("seat"),
                        player.get("name"),
                        player.get("stack"),
                        json.dumps(player.get("cards", [])),
                        player.get("action"),
                        player.get("bet_amount"),
                        player.get("is_active", True),
                        player.get("position"),
                    ),
                )

                # Update player profile if we have new information
                if player.get("name"):
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO player_profiles
                        (name, last_seen, avg_stack, total_hands)
                        VALUES (?, ?, ?, COALESCE((SELECT total_hands FROM player_profiles WHERE name = ?), 0) + 1)
                    """,
                        (
                            player["name"],
                            analysis.get("timestamp"),
                            player.get("stack"),
                            player["name"],
                        ),
                    )

            conn.commit()
            conn.close()

        except Exception as e:
            print(f"Error saving analysis to database: {e}")

    def analyze_table(self) -> Optional[Dict]:
        """Main method to analyze the current table state"""
        try:
            # Capture screenshot
            screenshot = self.capture_screenshot()
            if not screenshot:
                return None
            
            # Convert to OpenCV format for turn detection
            import cv2
            import numpy as np
            nparr = np.frombuffer(screenshot, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Check if it's the player's turn
            is_player_turn = self.detect_player_turn(image)
            
            if not is_player_turn:
                print(f"Not player's turn or no new decision (confidence: {self.turn_detection_confidence:.2f}), skipping analysis")
                return None
            
            # Check if game state has changed since last analysis
            if not self.has_state_changed(image):
                print("Game state unchanged since last analysis, skipping duplicate LLM call")
                return None
            
            print(f"Player's turn detected! Bet amount: {self.last_bet_amount}")
            
            # Play sound notification for debugging
            self.play_llm_notification()
            
            # Get existing player profiles
            player_profiles = self.get_player_profiles_from_db()
            
            # Create LLM prompt
            prompt = self.create_llm_prompt(player_profiles)
            
            # Encode image
            image_base64 = self.encode_image_for_llm(screenshot)
            
            # Call LLM
            llm_response = self.call_llm_vision_api(image_base64, prompt)
            if not llm_response:
                return None
            
            # Parse response (you'll need to extract JSON from LLM response)
            analysis = self.parse_llm_response(llm_response)
            if not analysis:
                return None
            
            # Add turn detection info
            analysis["turn_detection"] = {
                "is_player_turn": is_player_turn,
                "confidence": self.turn_detection_confidence,
                "detected_buttons": self.last_action_buttons,
                "bet_amount": self.last_bet_amount
            }
            
            # Update last analysis timestamp
            self.last_analysis_timestamp = datetime.now()
            
            # Save to database
            self.save_analysis_to_db(analysis)
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing table: {e}")
            return None

    def parse_llm_response(self, llm_response: Dict) -> Optional[Dict]:
        """Parse the LLM response to extract the JSON analysis"""
        # This will depend on your LLM API response format
        # Extract the JSON content from the response
        try:
            # Example for OpenAI:
            # content = llm_response["choices"][0]["message"]["content"]
            # return json.loads(content)

            # Placeholder - implement based on your LLM
            return None
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return None

    def run_continuous_analysis(self, interval_seconds: int = 1):
        """Run continuous table analysis"""
        print(
            f"Starting continuous table analysis (every {interval_seconds} second(s))"
        )
        print("Only analyzing when it's the player's turn...")

        while True:
            try:
                analysis = self.analyze_table()
                if analysis:
                    print(f"Analysis completed at {analysis.get('timestamp')}")
                    # You can add additional processing here
                else:
                    print("Skipped analysis - not player's turn")

                time.sleep(interval_seconds)

            except KeyboardInterrupt:
                print("Analysis stopped by user")
                break
            except Exception as e:
                print(f"Error in continuous analysis: {e}")
                time.sleep(interval_seconds)

    def create_game_state_hash(self, image) -> str:
        """Create a hash of the current game state to detect changes"""
        try:
            import hashlib
            
            # Convert image to grayscale and resize for consistent hashing
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (100, 100))  # Small size for fast hashing
            
            # Create hash from image data
            image_hash = hashlib.md5(resized.tobytes()).hexdigest()
            
            # Combine with bet amount for more specific state tracking
            bet_amount_str = str(self.last_bet_amount) if self.last_bet_amount else "none"
            state_hash = hashlib.md5(f"{image_hash}_{bet_amount_str}".encode()).hexdigest()
            
            return state_hash
            
        except Exception as e:
            print(f"Error creating game state hash: {e}")
            return "error"
    
    def has_state_changed(self, image) -> bool:
        """Check if the game state has changed since last analysis"""
        current_hash = self.create_game_state_hash(image)
        
        if self.last_analysis_hash != current_hash:
            self.last_analysis_hash = current_hash
            return True
        
        return False


if __name__ == "__main__":
    analyzer = TableAnalyzer()
    analyzer.run_continuous_analysis()
