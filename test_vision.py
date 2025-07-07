#!/usr/bin/env python3
# flake8: noqa: E402
"""
Test script to run the vision module on an image and print detected data.
"""
import os
import re
import sys
from typing import Optional

import cv2
import numpy as np
import pytesseract

# sys.path.insert must come after stdlib imports, before project imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from vision.card_recognizer import CardRecognizer, TemplateManager
from vision.number_reader import NumberReader
from vision.screen_capture import ImageProcessor, PokerStarsRegions, ScreenCapture


def test_image_processing(image_path: str) -> None:
    """Test image processing capabilities on a given image."""
    print(f"🔍 Testing vision module on: {image_path}")
    print("=" * 60)

    # Load the image
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found: {image_path}")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Could not load image: {image_path}")
        return

    print("✅ Image loaded successfully")
    print(f"   - Dimensions: {image.shape[1]}x{image.shape[0]} pixels")
    print(f"   - Channels: {image.shape[2] if len(image.shape) > 2 else 1}")
    print()

    # Test NumberReader (OCR for numbers)
    print("📊 Testing NumberReader (OCR for numbers)...")
    number_reader = NumberReader()

    # Try to read numbers from the entire image
    try:
        number = number_reader.read_number(image)
        if number is not None:
            print(f"   ✅ Detected number: {number}")
        else:
            print("   ⚠️  No numbers detected in the image")
    except Exception as e:
        print(f"   ❌ Error reading numbers: {e}")
    print()

    # Test ImageProcessor
    print("🖼️  Testing ImageProcessor...")
    processor = ImageProcessor()

    try:
        # Preprocess for OCR
        ocr_processed = processor.preprocess_for_ocr(image)
        print("   ✅ OCR preprocessing completed")

        # Enhance for card recognition
        card_enhanced = processor.enhance_card_image(image)
        print("   ✅ Card enhancement completed")

        # Detect text regions
        text_regions = processor.detect_text_regions(image)
        print(f"   ✅ Found {len(text_regions)} potential text regions")

        if text_regions:
            print("   📍 Text regions detected:")
            for i, (x, y, w, h) in enumerate(text_regions[:5]):  # Show first 5
                print(f"      Region {i+1}: ({x}, {y}) {w}x{h}")
            if len(text_regions) > 5:
                print(f"      ... and {len(text_regions) - 5} more regions")
    except Exception as e:
        print(f"   ❌ Error in image processing: {e}")
    print()

    # Test CardRecognizer (if templates exist)
    print("🃏 Testing CardRecognizer...")
    card_recognizer = CardRecognizer()

    if card_recognizer.templates:
        print(f"   ✅ Loaded {len(card_recognizer.templates)} card templates")

        # Try to recognize cards in the image
        try:
            # Define some test regions (you might need to adjust these)
            test_regions = [
                (50, 50, 100, 150),  # Example region 1
                (200, 50, 100, 150),  # Example region 2
            ]

            cards = card_recognizer.recognize_cards(image, test_regions)
            if cards:
                print(f"   ✅ Detected {len(cards)} cards:")
                for card in cards:
                    print(f"      - {card}")
            else:
                print("   ⚠️  No cards detected in test regions")
        except Exception as e:
            print(f"   ❌ Error in card recognition: {e}")
    else:
        print("   ⚠️  No card templates found. Run template creation first.")
    print()

    # Test TemplateManager
    print("📁 Testing TemplateManager...")
    template_manager = TemplateManager()

    try:
        validation = template_manager.validate_templates()
        total_templates = len(validation)
        existing_templates = sum(validation.values())

        print("   📊 Template status:")
        print(f"      - Total required: {total_templates}")
        print(f"      - Existing: {existing_templates}")
        print(f"      - Missing: {total_templates - existing_templates}")

        if existing_templates == 0:
            print("   💡 Tip: Create card templates first for better recognition")
    except Exception as e:
        print(f"   ❌ Error in template validation: {e}")
    print()

    # Save processed images for inspection
    print("💾 Saving processed images...")
    try:
        output_dir = "data/processed_images"
        os.makedirs(output_dir, exist_ok=True)

        # Save original
        cv2.imwrite(os.path.join(output_dir, "original.jpg"), image)

        # Save OCR processed
        ocr_processed = processor.preprocess_for_ocr(image)
        cv2.imwrite(os.path.join(output_dir, "ocr_processed.jpg"), ocr_processed)

        # Save card enhanced
        card_enhanced = processor.enhance_card_image(image)
        cv2.imwrite(os.path.join(output_dir, "card_enhanced.jpg"), card_enhanced)

        print(f"   ✅ Processed images saved to: {output_dir}/")
    except Exception as e:
        print(f"   ❌ Error saving processed images: {e}")
    print()

    print("🎯 Vision module test completed!")
    print("\n💡 Next steps:")
    print("   1. Check the processed images in data/processed_images/")
    print("   2. Create card templates for better card recognition")
    print("   3. Calibrate screen regions for your specific poker table layout")


def extract_pot_number_from_text(image: np.ndarray) -> Optional[float]:
    """Extract pot number from text like 'Pote: 80' using OCR."""
    text = pytesseract.image_to_string(image, config="--psm 7")
    print(f"🔍 OCR raw text: '{text}'")
    numbers = re.findall(r"\d+", text)
    if numbers:
        try:
            pot_value = float(numbers[0])
            print(f"✅ Extracted pot number: {pot_value}")
            return pot_value
        except ValueError:
            print(f"❌ Could not convert '{numbers[0]}' to number")
    else:
        print("❌ No numbers found in text")
    return None


def test_pot_size_from_image(image_path: str) -> None:
    print(f"\n🔎 Testing pot size detection on: {image_path}")
    # Load the image
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found: {image_path}")
        return
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Could not load image: {image_path}")
        return
    # Set up ScreenCapture and regions
    capture = ScreenCapture()
    PokerStarsRegions.setup_regions(capture)

    # ---
    def capture_region_from_image(region_name: str) -> Optional[np.ndarray]:
        info = capture.get_region_info(region_name)
        if info is None:
            print(f"❌ Region '{region_name}' not found.")
            return None
        x, y, w, h = info["x"], info["y"], info["width"], info["height"]
        region_img = image[y : y + h, x : x + w]
        # Save the region for visual inspection
        out_path = f"data/processed_images/{region_name}_region.jpg"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, region_img)
        print(f"🖼️  Saved {region_name} region to {out_path}")
        return region_img

    capture.capture_region = capture_region_from_image
    # Detect pot size using custom function
    pot_region = capture.capture_region("pot_size")
    if pot_region is not None:
        pot = extract_pot_number_from_text(pot_region)
        if pot is not None:
            print(f"✅ Detected pot size: {pot}")
        else:
            print("⚠️  Pot size could not be detected.")
    else:
        print("❌ Could not capture pot region")

    # Detect community cards region
    print("\n🃏 Testing community cards region...")
    community_region = capture.capture_region("community_cards")
    if community_region is not None:
        print("✅ Community cards region captured")
        # Try to detect any text in the community cards region
        try:
            community_text = pytesseract.image_to_string(
                community_region, config="--psm 6"
            )
            print(f"🔍 Community cards OCR text: '{community_text.strip()}'")
        except Exception as e:
            print(f"❌ Error reading community cards text: {e}")
    else:
        print("❌ Could not capture community cards region")

    # Detect hero hole cards region
    print("\n👤 Testing hero hole cards region...")
    hole_cards_region = capture.capture_region("hole_cards")
    if hole_cards_region is not None:
        print("✅ Hero hole cards region captured")
        # Try to detect any text in the hole cards region
        try:
            hole_cards_text = pytesseract.image_to_string(
                hole_cards_region, config="--psm 6"
            )
            print(f"🔍 Hero hole cards OCR text: '{hole_cards_text.strip()}'")
        except Exception as e:
            print(f"❌ Error reading hole cards text: {e}")
    else:
        print("❌ Could not capture hole cards region")


def main() -> None:
    """Main function to run vision tests."""
    if len(sys.argv) != 2:
        print("Usage: python test_vision.py <image_path>")
        print("Example: python test_vision.py data/poker_table.jpg")
        return

    image_path = sys.argv[1]
    test_image_processing(image_path)
    test_pot_size_from_image(image_path)


if __name__ == "__main__":
    main()
