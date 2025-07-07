"""
Poker Advisor: LLM integration for poker decision making.
"""
import json
import logging
from typing import Any, Dict, List, Optional

from openai import OpenAI

from core.poker_engine import Card, HandEvaluator

logger = logging.getLogger(__name__)


class PokerAdvisor:
    """Uses LLM to analyze poker situations and provide recommendations."""

    def __init__(self, api_key: str, model: str = "gpt-4") -> None:
        """Initialize the poker advisor with OpenAI API."""
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def analyze_situation(
        self,
        hole_cards: List[Card],
        community_cards: List[Card],
        pot_size: float,
        current_bet: float,
        player_stack: float,
        opponent_stack: float,
        position: str = "unknown",
        street: str = "unknown",
        action_history: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze poker situation and return structured recommendation.

        Returns:
            Dict with structured data for GUI processing:
            {
                "recommendation": "fold/call/raise/check",
                "confidence": 0.85,
                "reasoning": "Detailed explanation...",
                "expected_value": 15.5,
                "risk_level": "low/medium/high",
                "alternative_actions": [
                    {"action": "call", "confidence": 0.7, "reasoning": "..."},
                    {"action": "raise", "confidence": 0.3, "reasoning": "..."}
                ],
                "hand_strength": "strong/medium/weak",
                "pot_odds": 2.5,
                "position_advantage": "good/neutral/bad"
            }
        """
        try:
            # Prepare the prompt
            prompt = self._build_analysis_prompt(
                hole_cards=hole_cards,
                community_cards=community_cards,
                pot_size=pot_size,
                current_bet=current_bet,
                player_stack=player_stack,
                opponent_stack=opponent_stack,
                position=position,
                street=street,
                action_history=action_history,
            )

            # Get LLM response
            response = self._get_llm_response(prompt)

            # Parse and structure the response
            structured_response = self._parse_llm_response(response)

            # Add calculated metrics
            structured_response.update(
                self._calculate_metrics(
                    hole_cards=hole_cards,
                    community_cards=community_cards,
                    pot_size=pot_size,
                    current_bet=current_bet,
                    player_stack=player_stack,
                )
            )

            return structured_response

        except Exception as e:
            logger.error(f"Error analyzing poker situation: {e}")
            return self._get_fallback_response()

    def _build_analysis_prompt(
        self,
        hole_cards: List[Card],
        community_cards: List[Card],
        pot_size: float,
        current_bet: float,
        player_stack: float,
        opponent_stack: float,
        position: str,
        street: str,
        action_history: Optional[List[Dict]],
    ) -> str:
        """Build a comprehensive prompt for the LLM."""

        # Convert cards to readable format
        hole_str = " ".join([str(card) for card in hole_cards])
        community_str = (
            " ".join([str(card) for card in community_cards])
            if community_cards
            else "None"
        )

        # Build action history string
        action_str = ""
        if action_history:
            action_str = "\nAction History:\n"
            for action in action_history[-5:]:  # Last 5 actions
                player = action.get('player', 'Unknown')
                action_type = action.get('action', 'Unknown')
                amount = action.get('amount', 0)
                action_str += f"- {player}: {action_type} ${amount}\n"

        prompt = f"""
You are an expert poker player analyzing a Texas Hold'em situation. Provide a structured analysis and recommendation.

SITUATION:
- Your hole cards: {hole_str}
- Community cards: {community_str}
- Pot size: ${pot_size}
- Current bet to call: ${current_bet}
- Your stack: ${player_stack}
- Opponent stack: ${opponent_stack}
- Position: {position}
- Street: {street}
{action_str}

ANALYSIS REQUIREMENTS:
1. Evaluate hand strength relative to the board
2. Consider pot odds and implied odds
3. Factor in position and stack sizes
4. Analyze opponent tendencies from action history
5. Provide a clear recommendation with confidence level

RESPONSE FORMAT (JSON):
{{
    "recommendation": "fold/call/raise/check",
    "confidence": 0.85,
    "reasoning": "Detailed explanation of the decision...",
    "expected_value": 15.5,
    "risk_level": "low/medium/high",
    "alternative_actions": [
        {{
            "action": "call",
            "confidence": 0.7,
            "reasoning": "Why this could be good..."
        }}
    ],
    "hand_strength": "strong/medium/weak",
    "position_advantage": "good/neutral/bad"
}}

Provide only the JSON response, no additional text.
"""
        return prompt

    def _get_llm_response(self, prompt: str) -> str:
        """Get response from OpenAI LLM."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert poker player. "
                            "Provide analysis in the exact JSON format requested."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,  # Lower temperature for more consistent responses
                max_tokens=1000,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error getting LLM response: {e}")
            raise

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response and ensure it's properly structured."""
        try:
            # Try to extract JSON from response
            if response.startswith("```json"):
                response = response[7:-3]  # Remove markdown code blocks
            elif response.startswith("```"):
                response = response[3:-3]

            parsed = json.loads(response)

            # Ensure all required fields are present
            required_fields = {
                "recommendation": "fold",
                "confidence": 0.5,
                "reasoning": "Analysis not available",
                "expected_value": 0.0,
                "risk_level": "medium",
                "alternative_actions": [],
                "hand_strength": "medium",
                "position_advantage": "neutral",
            }

            for field, default_value in required_fields.items():
                if field not in parsed:
                    parsed[field] = default_value

            return parsed

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error parsing LLM response: {e}")
            return self._get_fallback_response()

    def _calculate_metrics(
        self,
        hole_cards: List[Card],
        community_cards: List[Card],
        pot_size: float,
        current_bet: float,
        player_stack: float,
    ) -> Dict[str, Any]:
        """Calculate additional metrics for the analysis."""
        metrics = {}

        # Calculate pot odds
        if current_bet > 0:
            metrics["pot_odds"] = pot_size / current_bet
        else:
            metrics["pot_odds"] = float("inf")

        # Calculate hand strength if we have community cards
        if len(community_cards) >= 3:
            try:
                hand_rank, hand_name = HandEvaluator.evaluate_hand(
                    hole_cards, community_cards
                )
                metrics["hand_rank"] = hand_rank
                metrics["hand_name"] = hand_name
            except Exception as e:
                logger.warning(f"Could not evaluate hand: {e}")
                metrics["hand_rank"] = 0
                metrics["hand_name"] = "Unknown"

        # Calculate stack-to-pot ratio
        metrics["stack_to_pot_ratio"] = (
            player_stack / pot_size if pot_size > 0 else float("inf")
        )

        return metrics

    def _get_fallback_response(self) -> Dict[str, Any]:
        """Return a fallback response when analysis fails."""
        return {
            "recommendation": "fold",
            "confidence": 0.5,
            "reasoning": "Unable to analyze situation. Consider folding for safety.",
            "expected_value": 0.0,
            "risk_level": "high",
            "alternative_actions": [
                {
                    "action": "fold",
                    "confidence": 0.8,
                    "reasoning": "When in doubt, fold.",
                }
            ],
            "hand_strength": "unknown",
            "position_advantage": "neutral",
            "pot_odds": 0.0,
            "stack_to_pot_ratio": 0.0,
        }


class PokerAnalysisResult:
    """Structured result from poker analysis for easy GUI processing."""

    def __init__(self, analysis_data: Dict[str, Any]) -> None:
        """Initialize with analysis data."""
        self.data = analysis_data

    @property
    def recommendation(self) -> str:
        """Get the main recommendation."""
        return self.data.get("recommendation", "fold")

    @property
    def confidence(self) -> float:
        """Get confidence level (0-1)."""
        return self.data.get("confidence", 0.5)

    @property
    def reasoning(self) -> str:
        """Get the reasoning for the recommendation."""
        return self.data.get("reasoning", "No reasoning available")

    @property
    def risk_level(self) -> str:
        """Get risk level."""
        return self.data.get("risk_level", "medium")

    @property
    def expected_value(self) -> float:
        """Get expected value."""
        return self.data.get("expected_value", 0.0)

    @property
    def hand_strength(self) -> str:
        """Get hand strength assessment."""
        return self.data.get("hand_strength", "unknown")

    @property
    def pot_odds(self) -> float:
        """Get pot odds."""
        return self.data.get("pot_odds", 0.0)

    @property
    def alternative_actions(self) -> List[Dict[str, Any]]:
        """Get alternative actions."""
        return self.data.get("alternative_actions", [])

    def to_gui_format(self) -> Dict[str, Any]:
        """Convert to GUI-friendly format."""
        return {
            "primary_action": {
                "action": self.recommendation,
                "confidence": self.confidence,
                "reasoning": self.reasoning,
                "risk_level": self.risk_level,
                "expected_value": self.expected_value,
            },
            "hand_info": {
                "strength": self.hand_strength,
                "pot_odds": self.pot_odds,
                "stack_to_pot_ratio": self.data.get("stack_to_pot_ratio", 0.0),
            },
            "alternatives": self.alternative_actions,
            "metrics": {
                "hand_rank": self.data.get("hand_rank", 0),
                "hand_name": self.data.get("hand_name", "Unknown"),
                "position_advantage": self.data.get("position_advantage", "neutral"),
            },
        }

    def get_action_color(self) -> str:
        """Get color for GUI based on recommendation."""
        colors = {
            "fold": "#ff4444",  # Red
            "call": "#ffaa00",  # Orange
            "raise": "#44ff44",  # Green
            "check": "#4444ff",  # Blue
        }
        return colors.get(self.recommendation, "#888888")

    def get_confidence_color(self) -> str:
        """Get color for confidence level."""
        if self.confidence >= 0.8:
            return "#44ff44"  # Green
        elif self.confidence >= 0.6:
            return "#ffaa00"  # Orange
        else:
            return "#ff4444"  # Red
