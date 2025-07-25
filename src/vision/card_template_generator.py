"""
Card Template Generator: Creates card templates from PokerStars screenshots.
"""
import os
from typing import List, Tuple

import cv2
import numpy as np

from core.poker_engine import Card


class CardTemplateGenerator:
    """Generates card templates from PokerStars screenshots."""

    def __init__(self, output_dir: str = "data/card_templates"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Card dimensions (will be calibrated)
        self.card_width = 40
        self.card_height = 60

        # Card regions for different positions
        self.card_regions = {
            "hole_cards": [(0, 0, 40, 60), (45, 0, 40, 60)],  # Hero's cards
            "community_cards": [
                (0, 0, 40, 60),
                (45, 0, 40, 60),
                (90, 0, 40, 60),
                (135, 0, 40, 60),
                (180, 0, 40, 60),
            ],  # Flop, turn, river
        }

    def extract_card_from_region(
        self, image: np.ndarray, region: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """Extract a card image from a specific region."""
        x, y, w, h = region
        card_img = image[y : y + h, x : x + w]

        # Preprocess for better template matching
        card_img = self.preprocess_card_image(card_img)
        return card_img

    def preprocess_card_image(self, card_img: np.ndarray) -> np.ndarray:
        """Preprocess card image for better recognition."""
        # Convert to grayscale
        if len(card_img.shape) == 3:
            gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = card_img.copy()

        # Enhance contrast
        gray = cv2.equalizeHist(gray)

        # Resize to standard size
        gray = cv2.resize(gray, (self.card_width, self.card_height))

        return gray

    def save_card_template(self, card: Card, template_img: np.ndarray) -> None:
        """Save a card template to disk."""
        filename = f"{card.rank.value}{card.suit.value}.png"
        filepath = os.path.join(self.output_dir, filename)
        cv2.imwrite(filepath, template_img)
        print(f"Saved template: {filename}")

    def generate_templates_from_screenshot(
        self, image_path: str, known_cards: List[Card]
    ) -> None:
        """Generate templates from a screenshot with known cards."""
        print(f"Generating templates from: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return

        # Extract cards from regions
        for i, card in enumerate(known_cards):
            if i < len(self.card_regions["hole_cards"]):
                region = self.card_regions["hole_cards"][i]
            elif i < len(self.card_regions["hole_cards"]) + len(
                self.card_regions["community_cards"]
            ):
                region = self.card_regions["community_cards"][
                    i - len(self.card_regions["hole_cards"])
                ]
            else:
                print(f"Too many cards: {len(known_cards)}")
                break

            card_img = self.extract_card_from_region(image, region)
            self.save_card_template(card, card_img)

    def create_all_templates(self) -> None:
        """Create templates for all 52 cards using reference images."""
        print("Creating templates for all 52 cards...")

        # This would require multiple screenshots with different cards
        # For now, we'll create a template creation guide
        self.create_template_guide()

    def create_template_guide(self) -> None:
        """Create a guide for manual template creation."""
        guide_path = os.path.join(self.output_dir, "TEMPLATE_GUIDE.md")

        guide_content = """# Card Template Creation Guide

## How to Create Card Templates

1. **Take screenshots** of PokerStars tables with visible cards
2. **Extract card images** from the screenshots
3. **Save as PNG** with naming convention: `{rank}{suit}.png`

## Naming Convention
- Rank: 2, 3, 4, 5, 6, 7, 8, 9, T, J, Q, K, A
- Suit: h (hearts), d (diamonds), c (clubs), s (spades)

## Examples
- `Ah.png` - Ace of Hearts
- `Kd.png` - King of Diamonds
- `7c.png` - 7 of Clubs
- `Ts.png` - 10 of Spades

## Card Dimensions
- Width: 40 pixels
- Height: 60 pixels
- Format: Grayscale PNG

## Tips
- Use high-contrast screenshots
- Ensure cards are clearly visible
- Avoid overlapping or obscured cards
- Use consistent lighting conditions
"""

        with open(guide_path, "w") as f:
            f.write(guide_content)

        print(f"Created template guide: {guide_path}")


def main() -> None:
    """Main function for template generation."""
    generator = CardTemplateGenerator()

    print("🎴 Card Template Generator")
    print("=" * 40)

    # Check if we have any existing templates
    existing_templates = len(
        [f for f in os.listdir(generator.output_dir) if f.endswith(".png")]
    )
    print(f"Existing templates: {existing_templates}")

    if existing_templates == 0:
        print("No templates found. Creating guide...")
        generator.create_template_guide()
        print("\nNext steps:")
        print("1. Take screenshots of PokerStars tables")
        print("2. Extract card images manually")
        print("3. Save them with the naming convention")
        print("4. Run the card recognizer again")
    else:
        print(f"Found {existing_templates} templates!")
        print("Templates are ready for use.")


if __name__ == "__main__":
    main()
