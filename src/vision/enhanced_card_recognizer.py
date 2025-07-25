"""
Enhanced Card Recognition System for PokerAI
Uses multiple techniques to distinguish actual cards from UI elements.
"""
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import cv2
import numpy as np
import pytesseract

from core.poker_engine import Card, Rank, Suit

logger = logging.getLogger(__name__)


@dataclass
class CardRegion:
    """Represents a detected card region with confidence metrics."""

    x: int
    y: int
    width: int
    height: int
    area: float
    confidence: float
    is_likely_card: bool
    card_value: Optional[Card] = None


class EnhancedCardRecognizer:
    """Enhanced card recognition with multiple validation techniques."""

    def __init__(self):
        """Initialize the enhanced card recognizer with optimized parameters."""
        # Card size constraints (updated for full card detection)
        self.min_card_area = 1500
        self.max_card_area = 25000

        # Aspect ratio constraints (playing cards are typically ~0.7)
        self.min_aspect_ratio = 0.4
        self.max_aspect_ratio = 1.2

        # Position constraints (cards should be in the top portion)
        self.max_y_position = 100  # Maximum y-position for cards

        # White background constraints (as percentages 0-100)
        self.min_white_percentage = 50
        self.max_white_percentage = 95

        # Multi-region detection parameters
        self.detection_region_width_ratio = 0.7  # 70% of card width
        self.detection_region_height_ratio = 0.5  # 50% of card height

        # Detection parameters (no longer using templates)

    def detect_card_regions(self, image: np.ndarray) -> List[CardRegion]:
        """
        Detect potential card regions using improved morphological analysis.

        Args:
            image: Input image (BGR format)

        Returns:
            List of CardRegion objects with confidence scores
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Step 1: Create binary image
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Step 2: Morphological operations to connect nearby white pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Step 3: Find contours
        contours, _ = cv2.findContours(
            morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        card_regions = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by area
            if area < self.min_card_area or area > self.max_card_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0

            # Filter by aspect ratio
            if (
                aspect_ratio < self.min_aspect_ratio
                or aspect_ratio > self.max_aspect_ratio
            ):
                continue

            # Filter by y-position
            if y > self.max_y_position:
                continue

            # Step 3: Analyze region characteristics
            region_analysis = self._analyze_region(image, x, y, w, h)

            # Step 4: Calculate confidence score
            confidence = self._calculate_confidence(region_analysis, area, aspect_ratio)

            # Step 5: Determine if it's likely a card
            is_likely_card = confidence > 0.5

            card_region = CardRegion(
                x=x,
                y=y,
                width=w,
                height=h,
                area=area,
                confidence=confidence,
                is_likely_card=is_likely_card,
            )

            card_regions.append(card_region)

        # Sort by confidence (highest first)
        card_regions.sort(key=lambda r: r.confidence, reverse=True)

        logger.info(f"Detected {len(card_regions)} potential card regions")
        return card_regions

    def _analyze_region(
        self, image: np.ndarray, x: int, y: int, w: int, h: int
    ) -> Dict[str, Any]:
        """Analyze a region for card-like characteristics."""
        region = image[y : y + h, x : x + w]
        gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        # Color analysis
        mean_color = np.mean(region, axis=(0, 1))
        std_color = np.std(region, axis=(0, 1))

        # White pixel analysis
        white_pixels = np.sum(gray_region > 200)
        total_pixels = gray_region.size
        white_percentage = (white_pixels / total_pixels) * 100

        # Edge density analysis
        edges = cv2.Canny(gray_region, 50, 150)
        edge_density = np.sum(edges > 0) / total_pixels

        # Texture analysis (standard deviation of grayscale)
        texture_variance = np.var(gray_region)

        # Check for card-like patterns (corners, borders)
        corners = cv2.goodFeaturesToTrack(gray_region, 4, 0.01, 10)
        corner_count = len(corners) if corners is not None else 0

        return {
            "mean_color": mean_color,
            "std_color": std_color,
            "white_percentage": white_percentage,
            "edge_density": edge_density,
            "texture_variance": texture_variance,
            "corner_count": corner_count,
            "region": region,
            "gray_region": gray_region,
        }

    def _calculate_confidence(
        self, analysis: Dict[str, Any], area: float, aspect_ratio: float
    ) -> float:
        """Calculate confidence score for a region being a card."""
        confidence = 0.0

        # White background score (cards should have white background)
        white_score = 0.0
        white_pct = analysis["white_percentage"]
        if self.min_white_percentage <= white_pct <= self.max_white_percentage:
            white_score = 1.0 - abs(white_pct - 60) / 30  # Peak at 60%
        confidence += white_score * 0.3

        # Edge density score (cards should have clear edges)
        edge_score = min(analysis["edge_density"] * 100, 1.0)
        confidence += edge_score * 0.2

        # Texture variance score (cards should have some texture, not uniform)
        texture_score = min(analysis["texture_variance"] / 1000, 1.0)
        confidence += texture_score * 0.2

        # Corner detection score (cards should have 4 corners)
        corner_score = 1.0 - abs(analysis["corner_count"] - 4) / 4
        corner_score = max(0, corner_score)
        confidence += corner_score * 0.15

        # Aspect ratio score (cards should have reasonable proportions)
        aspect_score = 1.0 - abs(aspect_ratio - 0.7) / 0.3  # Peak at 0.7
        aspect_score = max(0, aspect_score)
        confidence += aspect_score * 0.15

        return min(confidence, 1.0)

    def recognize_cards(self, image: np.ndarray) -> List[Card]:
        """
        Recognize cards from detected regions using multi-region OCR and color analysis.

        Args:
            image: Input image (BGR format)

        Returns:
            List of recognized Card objects
        """
        card_regions = self.detect_card_regions(image)

        if not card_regions:
            return []

        # Extract multiple detection regions for each card
        detection_regions = self._extract_detection_regions(card_regions)

        # Recognize content from all regions
        all_results = []
        for region in detection_regions:
            rank, suit, confidence = self._recognize_card_content(image, region)
            if rank and suit:
                all_results.append(
                    {
                        "rank": rank,
                        "suit": suit,
                        "confidence": confidence,
                        "card_number": region.get("card_number", 0),
                        "region_type": region.get("region_type", "unknown"),
                    }
                )

        # Combine results and convert to Card objects
        final_results = self._combine_card_results(all_results)
        recognized_cards = []

        for rank, suit, confidence in final_results:
            if rank and suit:
                try:
                    card = self._create_card_from_rank_suit(rank, suit)
                    if card:
                        recognized_cards.append(card)
                        logger.info(
                            f"Recognized card: {card} (confidence: {confidence:.2f})"
                        )
                except (ValueError, TypeError):
                    logger.warning(f"Invalid card combination: {rank}{suit}")

        return recognized_cards

    def _extract_detection_regions(
        self, card_regions: List[CardRegion]
    ) -> List[Dict[str, Any]]:
        """Extract multiple detection regions for each card (upper left and upper right corners)."""

        all_regions = []

        for i, region in enumerate(card_regions):
            x, y = region.x, region.y
            w, h = region.width, region.height

            # Calculate detection region dimensions
            corner_w = int(w * self.detection_region_width_ratio)
            corner_h = int(h * self.detection_region_height_ratio)

            # Upper left corner region
            upper_left = {
                "x": x,
                "y": y,
                "width": corner_w,
                "height": corner_h,
                "confidence": region.confidence,
                "card_number": i + 1,
                "region_type": "upper_left",
            }

            # Upper right corner region
            upper_right = {
                "x": x + w - corner_w,
                "y": y,
                "width": corner_w,
                "height": corner_h,
                "confidence": region.confidence,
                "card_number": i + 1,
                "region_type": "upper_right",
            }

            all_regions.extend([upper_left, upper_right])

        return all_regions

    def _recognize_card_content(
        self, image: np.ndarray, region: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[str], float]:
        """Recognize card rank and suit from a detection region using OCR and color analysis."""

        x, y = region["x"], region["y"]
        w, h = region["width"], region["height"]

        # Extract region
        region_img = image[y : y + h, x : x + w]

        if region_img.size == 0:
            return None, None, 0.0

        # Convert to different color spaces
        gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(region_img, cv2.COLOR_BGR2HSV)

        # Recognize rank and suit
        rank = self._recognize_rank(gray)
        suit = self._recognize_suit(hsv)

        # Calculate confidence
        confidence = 0.0
        if rank is not None:
            confidence += 0.4
        if suit is not None:
            confidence += 0.4
        if rank is not None and suit is not None:
            confidence += 0.2

        return rank, suit, confidence

    def _recognize_rank(self, gray_region: np.ndarray) -> Optional[str]:
        """Recognize card rank using OCR with multiple preprocessing techniques."""

        # Try multiple preprocessing techniques
        techniques = []

        # Technique 1: Basic threshold
        _, binary1 = cv2.threshold(gray_region, 127, 255, cv2.THRESH_BINARY)
        techniques.append(binary1)

        # Technique 2: Otsu threshold
        _, binary2 = cv2.threshold(
            gray_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        techniques.append(binary2)

        # Technique 3: CLAHE + Otsu
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray_region)
        _, binary3 = cv2.threshold(
            clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        techniques.append(binary3)

        # OCR configuration for playing cards
        config = "--oem 3 --psm 8 -c tessedit_char_whitelist=23456789TJQKA"

        best_result = None
        best_confidence = 0

        for binary in techniques:
            try:
                # Use pytesseract directly since we don't have self._ocr_engine
                import pytesseract

                text = pytesseract.image_to_string(binary, config=config).strip()
                text = "".join(c for c in text if c.isalnum())

                if text:
                    confidence = 0

                    # Map common OCR mistakes and validate
                    rank_mapping = {
                        "T": "10",  # Ten
                        "J": "J",  # Jack
                        "Q": "Q",  # Queen
                        "K": "K",  # King
                        "A": "A",  # Ace
                    }

                    if text in rank_mapping:
                        result = rank_mapping[text]
                        confidence = 1.0
                    elif text.isdigit() and 2 <= int(text) <= 10:
                        result = text
                        confidence = 1.0
                    elif (
                        len(text) == 2 and text[0] == text[1]
                    ):  # Repeated digits like "66"
                        result = text[0]
                        confidence = 0.9
                    elif (
                        len(text) >= 3
                        and text[:2].isdigit()
                        and 2 <= int(text[:2]) <= 10
                    ):  # "323" -> "3"
                        result = text[0]
                        confidence = 0.8
                    else:
                        continue

                    if confidence > best_confidence:
                        best_result = result
                        best_confidence = confidence

            except Exception as e:
                logger.debug(f"OCR technique failed: {e}")
                continue

        return best_result

    def _recognize_suit(self, hsv_region: np.ndarray) -> Optional[str]:
        """Recognize card suit using improved color analysis."""

        # Red suits (Hearts, Diamonds)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        red_mask1 = cv2.inRange(hsv_region, lower_red1, upper_red1)

        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        red_mask2 = cv2.inRange(hsv_region, lower_red2, upper_red2)

        red_pixels = np.sum(red_mask1) + np.sum(red_mask2)

        # Blue (Diamonds) - Improved detection
        lower_blue1 = np.array([100, 50, 50])
        upper_blue1 = np.array([130, 255, 255])
        blue_mask1 = cv2.inRange(hsv_region, lower_blue1, upper_blue1)

        lower_blue2 = np.array([110, 30, 30])
        upper_blue2 = np.array([140, 255, 255])
        blue_mask2 = cv2.inRange(hsv_region, lower_blue2, upper_blue2)

        blue_pixels = np.sum(blue_mask1) + np.sum(blue_mask2)

        # Green (Clubs)
        lower_green = np.array([40, 100, 100])
        upper_green = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv_region, lower_green, upper_green)
        green_pixels = np.sum(green_mask)

        # Black (Spades)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])
        black_mask = cv2.inRange(hsv_region, lower_black, upper_black)
        black_pixels = np.sum(black_mask)

        # Determine suit based on dominant color
        total_pixels = hsv_region.shape[0] * hsv_region.shape[1]
        color_threshold = total_pixels * 0.005  # 0.5% threshold

        if blue_pixels > color_threshold:
            return "diamonds"  # Diamond (blue)
        elif red_pixels > color_threshold:
            return "hearts"  # Heart (red)
        elif green_pixels > color_threshold:
            return "clubs"  # Club (green)
        elif black_pixels > color_threshold:
            return "spades"  # Spade (black)
        else:
            return None

    def _combine_card_results(
        self, all_results: List[Dict[str, Any]]
    ) -> List[Tuple[Optional[str], Optional[str], float]]:
        """Combine results from multiple regions to get the best detection for each card."""

        # Group results by card number
        card_results = {}

        for result in all_results:
            rank, suit, confidence = (
                result["rank"],
                result["suit"],
                result["confidence"],
            )
            card_num = result["card_number"]
            region_type = result["region_type"]

            if card_num not in card_results:
                card_results[card_num] = []

            card_results[card_num].append(
                {
                    "rank": rank,
                    "suit": suit,
                    "confidence": confidence,
                    "region_type": region_type,
                }
            )

        # For each card, pick the best result
        final_results = []

        for card_num in sorted(card_results.keys()):
            results = card_results[card_num]

            # Find the best result (highest confidence with both rank and suit)
            best_result = None
            best_confidence = 0

            for result in results:
                rank = result["rank"]
                suit = result["suit"]
                confidence = result["confidence"]

                # Prefer results with both rank and suit
                if rank and suit:
                    if confidence > best_confidence:
                        best_result = (rank, suit, confidence)
                        best_confidence = confidence
                elif rank and not best_result:  # Fallback to rank-only
                    best_result = (rank, None, confidence)
                    best_confidence = confidence
                elif suit and not best_result:  # Fallback to suit-only
                    best_result = (None, suit, confidence)
                    best_confidence = confidence

            if best_result:
                final_results.append(best_result)
            else:
                final_results.append((None, None, 0.0))

        return final_results

    def _create_card_from_rank_suit(self, rank: str, suit: str) -> Optional[Card]:
        """Create a Card object from rank and suit strings."""

        # Parse rank
        rank_map = {
            "2": Rank.TWO,
            "3": Rank.THREE,
            "4": Rank.FOUR,
            "5": Rank.FIVE,
            "6": Rank.SIX,
            "7": Rank.SEVEN,
            "8": Rank.EIGHT,
            "9": Rank.NINE,
            "10": Rank.TEN,
            "T": Rank.TEN,
            "J": Rank.JACK,
            "Q": Rank.QUEEN,
            "K": Rank.KING,
            "A": Rank.ACE,
        }

        # Parse suit
        suit_map = {
            "hearts": Suit.HEARTS,
            "diamonds": Suit.DIAMONDS,
            "clubs": Suit.CLUBS,
            "spades": Suit.SPADES,
        }

        if rank in rank_map and suit in suit_map:
            return Card(rank_map[rank], suit_map[suit])

        return None

    def get_detection_summary(self, image: np.ndarray) -> Dict[str, Any]:
        """Get a comprehensive summary of card detection results."""
        card_regions = self.detect_card_regions(image)
        recognized_cards = self.recognize_cards(image)

        return {
            "total_regions_detected": len(card_regions),
            "likely_card_regions": len([r for r in card_regions if r.is_likely_card]),
            "recognized_cards": recognized_cards,
            "recognition_confidence": np.mean([r.confidence for r in card_regions])
            if card_regions
            else 0,
            "region_details": [
                {
                    "position": (r.x, r.y),
                    "size": (r.width, r.height),
                    "confidence": r.confidence,
                    "is_likely_card": r.is_likely_card,
                    "card_value": str(r.card_value) if r.card_value else None,
                }
                for r in card_regions[:5]  # Top 5 regions
            ],
        }
