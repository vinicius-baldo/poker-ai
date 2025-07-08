#!/usr/bin/env python3
"""
Button Detection Calibration Script
Helps you calibrate the regions where action buttons appear in your poker client.
"""

import json
from pathlib import Path

import cv2


class ButtonDetectionCalibrator:
    def __init__(self):
        self.button_regions = {
            "fold": {"x": 300, "y": 550, "w": 80, "h": 30},
            "check_call": {"x": 400, "y": 550, "w": 80, "h": 30},
            "bet_raise": {"x": 500, "y": 550, "w": 80, "h": 30},
            "all_in": {"x": 600, "y": 550, "w": 80, "h": 30},
        }
        self.colors = {
            "fold": (0, 0, 255),  # Red
            "check_call": (0, 255, 0),  # Green
            "bet_raise": (255, 0, 0),  # Blue
            "all_in": (255, 255, 0),  # Cyan
        }

    def calibrate_from_image(
        self, image_path: str, output_path: str = "button_calibration.png"
    ):
        """Calibrate button regions from a screenshot"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return

        # Create a copy for drawing
        calibration_image = image.copy()

        # Draw current button regions
        for button_name, region in self.button_regions.items():
            x, y, w, h = region["x"], region["y"], region["w"], region["h"]
            color = self.colors[button_name]

            # Draw rectangle
            cv2.rectangle(calibration_image, (x, y), (x + w, y + h), color, 2)

            # Add label
            cv2.putText(
                calibration_image,
                button_name,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

            # Analyze brightness in region
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            button_region = gray[y : y + h, x : x + w]
            mean_brightness = cv2.mean(button_region)[0]

            # Add brightness info
            brightness_text = f"{mean_brightness:.1f}"
            cv2.putText(
                calibration_image,
                brightness_text,
                (x, y + h + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )

        # Save calibration image
        cv2.imwrite(output_path, calibration_image)
        print(f"Calibration image saved to {output_path}")

        # Print current regions
        print("\nCurrent button regions:")
        for button_name, region in self.button_regions.items():
            print(
                f"  {button_name}: x={region['x']}, y={region['y']}, w={region['w']}, h={region['h']}"
            )

        # Test turn detection
        self.test_turn_detection(image)

    def test_turn_detection(self, image):
        """Test turn detection on the image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        active_buttons = []
        for button_name, region in self.button_regions.items():
            x, y, w, h = region["x"], region["y"], region["w"], region["h"]
            button_region = gray[y : y + h, x : x + w]

            mean_brightness = cv2.mean(button_region)[0]
            is_active = mean_brightness > 150

            if is_active:
                active_buttons.append(button_name)

            print(
                f"  {button_name}: brightness={mean_brightness:.1f}, active={is_active}"
            )

        is_turn = len(active_buttons) > 0
        confidence = len(active_buttons) / len(self.button_regions)

        print("\nTurn detection result:")
        print(f"  Is player's turn: {is_turn}")
        print(f"  Confidence: {confidence:.2f}")
        print(f"  Active buttons: {active_buttons}")

    def update_button_region(self, button_name: str, x: int, y: int, w: int, h: int):
        """Update a button region"""
        if button_name in self.button_regions:
            self.button_regions[button_name] = {"x": x, "y": y, "w": w, "h": h}
            print(f"Updated {button_name} region: x={x}, y={y}, w={w}, h={h}")
        else:
            print(f"Unknown button: {button_name}")

    def save_config(self, config_path: str = "config/button_regions.json"):
        """Save button regions to config file"""
        config = {
            "button_regions": self.button_regions,
            "brightness_threshold": 150,
            "description": "Regions for detecting action buttons in poker client",
        }

        # Create directory if it doesn't exist
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"Button regions saved to {config_path}")

    def load_config(self, config_path: str = "config/button_regions.json"):
        """Load button regions from config file"""
        try:
            with open(config_path, "r") as f:
                config = json.load(f)

            self.button_regions = config.get("button_regions", self.button_regions)
            print(f"Loaded button regions from {config_path}")

        except FileNotFoundError:
            print(f"Config file {config_path} not found, using defaults")
        except Exception as e:
            print(f"Error loading config: {e}")


def main():
    """Main function for button calibration"""
    calibrator = ButtonDetectionCalibrator()

    # Try to load existing config
    calibrator.load_config()

    # Test with sample image
    test_image = "imagem_teste.png"  # or your screenshot
    if Path(test_image).exists():
        print(f"🎯 Calibrating button detection with {test_image}")
        print("=" * 50)

        calibrator.calibrate_from_image(test_image)

        print("\n📝 Instructions:")
        print("1. Look at the calibration image to see if button regions are correct")
        print("2. If regions are wrong, update them using:")
        print("   calibrator.update_button_region('fold', x, y, w, h)")
        print("3. Save the configuration:")
        print("   calibrator.save_config()")
        print("4. Test again with the updated regions")

        # Example of how to update regions:
        print("\n💡 Example updates:")
        print("# calibrator.update_button_region('fold', 320, 560, 70, 25)")
        print("# calibrator.update_button_region('check_call', 420, 560, 70, 25)")
        print("# calibrator.save_config()")

    else:
        print(f"❌ Test image {test_image} not found")
        print(
            "Please provide a screenshot of your poker table with action buttons visible"
        )


if __name__ == "__main__":
    main()
