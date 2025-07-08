#!/usr/bin/env python3
"""
Slider Detection Calibration Script
Helps you calibrate the region where the bet slider appears in your poker client.
"""

import json
from pathlib import Path

import cv2
import pytesseract


class ButtonDetectionCalibrator:
    def __init__(self):
        # Start with a single region in the bottom right corner
        self.slider_region = {"x": 1385, "y": 1250, "w": 120, "h": 40}  # Moved right 10px
        self.color = (0, 255, 255)  # Yellow

    def calibrate_from_image(self, image_path: str, output_path: str = "slider_calibration.png"):
        """Calibrate slider region from a screenshot"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return

        # Create a copy for drawing
        calibration_image = image.copy()

        # Draw slider region
        x, y, w, h = self.slider_region["x"], self.slider_region["y"], self.slider_region["w"], self.slider_region["h"]
        color = self.color
        cv2.rectangle(calibration_image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(calibration_image, "slider", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Analyze brightness in region
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        slider_region = gray[y : y + h, x : x + w]
        mean_brightness = cv2.mean(slider_region)[0]
        brightness_text = f"{mean_brightness:.1f}"
        cv2.putText(calibration_image, brightness_text, (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Save calibration image
        cv2.imwrite(output_path, calibration_image)
        print(f"Calibration image saved to {output_path}")
        print(f"Current slider region: x={x}, y={y}, w={w}, h={h}")
        print(f"Slider region brightness: {mean_brightness:.1f}")

        # Test slider detection and OCR
        self.test_slider_detection(image)

    def test_slider_detection(self, image):
        """Test slider detection and OCR on the image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        x, y, w, h = self.slider_region["x"], self.slider_region["y"], self.slider_region["w"], self.slider_region["h"]
        slider_region = gray[y : y + h, x : x + w]
        mean_brightness = cv2.mean(slider_region)[0]
        is_slider_present = mean_brightness > 80  # You may adjust this threshold
        
        # Try to read text from the region
        try:
            # Preprocess the region for better OCR
            # Resize for better recognition
            slider_region_resized = cv2.resize(slider_region, (w*2, h*2))
            
            # Apply threshold to make text more readable
            _, thresh = cv2.threshold(slider_region_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # OCR configuration for numbers
            config = '--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.'
            
            # Read text
            text = pytesseract.image_to_string(thresh, config=config).strip()
            
            print(f"\nSlider detection result:")
            print(f"  Slider present: {is_slider_present}")
            print(f"  Brightness: {mean_brightness:.1f}")
            print(f"  Detected text: '{text}'")
            
            # Try to extract number
            if text:
                # Clean up the text and try to extract a number
                import re
                numbers = re.findall(r'\d+\.?\d*', text)
                if numbers:
                    print(f"  Extracted number: {numbers[0]}")
                else:
                    print(f"  No number found in text")
            else:
                print(f"  No text detected")
                
        except Exception as e:
            print(f"\nSlider detection result:")
            print(f"  Slider present: {is_slider_present}")
            print(f"  Brightness: {mean_brightness:.1f}")
            print(f"  OCR error: {e}")

    def update_slider_region(self, x: int, y: int, w: int, h: int):
        self.slider_region = {"x": x, "y": y, "w": w, "h": h}
        print(f"Updated slider region: x={x}, y={y}, w={w}, h={h}")

    def save_config(self, config_path: str = "config/slider_region.json"):
        config = {
            "slider_region": self.slider_region,
            "brightness_threshold": 80,
            "description": "Region for detecting bet slider in poker client"
        }
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Slider region saved to {config_path}")

    def load_config(self, config_path: str = "config/slider_region.json"):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            self.slider_region = config.get("slider_region", self.slider_region)
            print(f"Loaded slider region from {config_path}")
        except FileNotFoundError:
            print(f"Config file {config_path} not found, using defaults")
        except Exception as e:
            print(f"Error loading config: {e}")


def main():
    calibrator = ButtonDetectionCalibrator()
    calibrator.load_config()
    test_image = "imagem_tela.png"
    if Path(test_image).exists():
        print(f"🎯 Calibrating slider detection with {test_image}")
        print("=" * 50)
        calibrator.calibrate_from_image(test_image)
        print("\n📝 Instructions:")
        print("1. Look at the calibration image to see if the slider region is correct")
        print("2. If the region is wrong, update it using:")
        print("   calibrator.update_slider_region(x, y, w, h)")
        print("3. Save the configuration:")
        print("   calibrator.save_config()")
        print("4. Test again with the updated region")
        print("\n💡 Example update:")
        print("# calibrator.update_slider_region(900, 650, 120, 40)")
        print("# calibrator.save_config()")
    else:
        print(f"❌ Test image {test_image} not found")
        print("Please provide a screenshot of your poker table with the bet slider visible")


if __name__ == "__main__":
    main()
