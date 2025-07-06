"""
Card recognition using template matching for PokerStars.
"""
import cv2
import numpy as np
import os
from typing import List, Tuple, Optional, Dict
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.poker_engine import Card, Rank, Suit
import logging

logger = logging.getLogger(__name__)


class CardTemplate:
    """Represents a card template for matching."""
    
    def __init__(self, rank: Rank, suit: Suit, template: np.ndarray, confidence: float = 0.8):
        self.rank = rank
        self.suit = suit
        self.template = template
        self.confidence = confidence
        self.card = Card(rank, suit)
    
    def __str__(self) -> str:
        return str(self.card)


class CardRecognizer:
    """Recognizes cards using template matching."""
    
    def __init__(self, template_dir: str = "data/card_templates"):
        self.template_dir = template_dir
        self.templates: Dict[str, CardTemplate] = {}
        self.load_templates()
    
    def load_templates(self):
        """Load card templates from directory."""
        if not os.path.exists(self.template_dir):
            logger.warning(f"Template directory {self.template_dir} not found. Creating it.")
            os.makedirs(self.template_dir, exist_ok=True)
            return
        
        # Load all template images
        for filename in os.listdir(self.template_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(self.template_dir, filename)
                try:
                    template_img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                    if template_img is not None:
                        # Parse filename to get rank and suit
                        card_name = filename.split('.')[0]  # Remove extension
                        card = self._parse_card_name(card_name)
                        if card:
                            template = CardTemplate(card.rank, card.suit, template_img)
                            self.templates[str(card)] = template
                            logger.debug(f"Loaded template for {card}")
                except Exception as e:
                    logger.error(f"Failed to load template {filename}: {e}")
        
        logger.info(f"Loaded {len(self.templates)} card templates")
    
    def _parse_card_name(self, card_name: str) -> Optional[Card]:
        """Parse card name from filename (e.g., 'Ah' -> Ace of Hearts)."""
        if len(card_name) != 2:
            return None
        
        rank_char, suit_char = card_name[0], card_name[1]
        
        # Parse rank
        rank_map = {
            '2': Rank.TWO, '3': Rank.THREE, '4': Rank.FOUR, '5': Rank.FIVE,
            '6': Rank.SIX, '7': Rank.SEVEN, '8': Rank.EIGHT, '9': Rank.NINE,
            'T': Rank.TEN, 'J': Rank.JACK, 'Q': Rank.QUEEN, 'K': Rank.KING, 'A': Rank.ACE
        }
        
        # Parse suit
        suit_map = {
            'h': Suit.HEARTS, 'd': Suit.DIAMONDS, 'c': Suit.CLUBS, 's': Suit.SPADES
        }
        
        if rank_char in rank_map and suit_char in suit_map:
            return Card(rank_map[rank_char], suit_map[suit_char])
        
        return None
    
    def create_template(self, card: Card, image: np.ndarray, region: Tuple[int, int, int, int]):
        """Create a template for a card from an image region."""
        x, y, w, h = region
        card_img = image[y:y+h, x:x+w]
        
        # Convert to grayscale
        gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
        
        # Save template
        filename = f"{card}.png"
        filepath = os.path.join(self.template_dir, filename)
        cv2.imwrite(filepath, gray)
        
        # Add to templates
        template = CardTemplate(card.rank, card.suit, gray)
        self.templates[str(card)] = template
        
        logger.info(f"Created template for {card}")
    
    def recognize_cards(self, image: np.ndarray, regions: List[Tuple[int, int, int, int]]) -> List[Card]:
        """Recognize cards in the given image regions."""
        cards = []
        
        for region in regions:
            card = self.recognize_card(image, region)
            if card:
                cards.append(card)
        
        return cards
    
    def recognize_card(self, image: np.ndarray, region: Tuple[int, int, int, int]) -> Optional[Card]:
        """Recognize a single card in the given region."""
        x, y, w, h = region
        card_img = image[y:y+h, x:x+w]
        
        # Convert to grayscale
        gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
        
        best_match = None
        best_score = 0
        
        # Try to match against all templates
        for template in self.templates.values():
            # Resize template to match card image size
            resized_template = cv2.resize(template.template, (w, h))
            
            # Use template matching
            result = cv2.matchTemplate(gray, resized_template, cv2.TM_CCOEFF_NORMED)
            score = np.max(result)
            
            if score > best_score and score > template.confidence:
                best_score = score
                best_match = template.card
        
        if best_match:
            logger.debug(f"Recognized card {best_match} with confidence {best_score:.3f}")
            return best_match
        
        logger.debug(f"No card recognized in region {region}")
        return None
    
    def recognize_hole_cards(self, image: np.ndarray) -> List[Card]:
        """Recognize hole cards from the captured image."""
        # Define regions for hole cards (these will need calibration)
        card_width = 30
        card_height = 40
        card_spacing = 5
        
        # Assuming hole cards are side by side
        regions = [
            (0, 0, card_width, card_height),
            (card_width + card_spacing, 0, card_width, card_height)
        ]
        
        return self.recognize_cards(image, regions)
    
    def recognize_community_cards(self, image: np.ndarray) -> List[Card]:
        """Recognize community cards from the captured image."""
        # Define regions for community cards (5 cards in a row)
        card_width = 30
        card_height = 40
        card_spacing = 5
        
        regions = []
        for i in range(5):
            x = i * (card_width + card_spacing)
            regions.append((x, 0, card_width, card_height))
        
        return self.recognize_cards(image, regions)
    
    def calibrate_card_regions(self, image: np.ndarray, known_cards: List[Card]) -> List[Tuple[int, int, int, int]]:
        """
        Calibrate card regions based on known cards.
        This would be used during setup to determine exact card positions.
        """
        # TODO: Implement automatic region calibration
        logger.info("Card region calibration not yet implemented")
        return []


class TemplateManager:
    """Manages card templates and provides utilities for template creation."""
    
    def __init__(self, template_dir: str = "data/card_templates"):
        self.template_dir = template_dir
        os.makedirs(template_dir, exist_ok=True)
    
    def create_all_templates(self, reference_image: str):
        """
        Create templates for all cards from a reference image.
        This would be used during initial setup.
        """
        logger.info("Template creation from reference image not yet implemented")
        # TODO: Implement automatic template creation
        pass
    
    def validate_templates(self) -> Dict[str, bool]:
        """Validate that all required templates exist."""
        required_cards = []
        for rank in Rank:
            for suit in Suit:
                card = Card(rank, suit)
                required_cards.append(str(card))
        
        validation = {}
        for card_name in required_cards:
            filepath = os.path.join(self.template_dir, f"{card_name}.png")
            validation[card_name] = os.path.exists(filepath)
        
        missing = [card for card, exists in validation.items() if not exists]
        if missing:
            logger.warning(f"Missing templates for: {missing}")
        
        return validation 