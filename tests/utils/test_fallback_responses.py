"""
Tests for Fallback Responses

Validates that fallback responses are valid and safe.
"""

import pytest
from src.utils.fallback_responses import (
    get_fallback_classification,
    get_fallback_strategy,
    get_fallback_message,
    get_fallback_message_spanish,
    get_fallback_message_english,
)
from src.models.classifier_response import Intent, UrgencyLevel
from src.models.director_response import StrategicAction
from src.models.intelligence import Sentiment


class TestFallbackClassification:
    """Tests for classifier fallback."""

    def test_returns_unclear_intent(self):
        """Should return unclear intent to avoid assumptions."""
        result = get_fallback_classification()
        assert result.intent == Intent.UNCLEAR

    def test_zero_confidence(self):
        """Should have zero confidence on all classifications."""
        result = get_fallback_classification()
        assert result.intent_confidence == 0.0
        assert result.topic_confidence == 0.0
        assert result.urgency_confidence == 0.0

    def test_flags_for_human_review(self):
        """Should flag for human escalation."""
        result = get_fallback_classification()
        assert result.requires_human_escalation is True

    def test_neutral_sentiment(self):
        """Should assume neutral sentiment."""
        result = get_fallback_classification()
        assert result.sentiment == Sentiment.NEUTRAL

    def test_is_valid_model(self):
        """Should be a valid ClassifierResponse."""
        result = get_fallback_classification()
        # Pydantic validation happens on construction
        assert result.reasoning is not None
        assert len(result.reasoning) > 0


class TestFallbackStrategy:
    """Tests for director fallback."""

    def test_returns_help_action(self):
        """Should return HELP action (non-aggressive)."""
        result = get_fallback_strategy()
        assert result.action == StrategicAction.HELP

    def test_no_stage_transition(self):
        """Should not suggest stage changes."""
        result = get_fallback_strategy()
        assert result.suggested_stage_transition is None

    def test_has_valid_message_strategy(self):
        """Should have complete message strategy."""
        result = get_fallback_strategy()
        assert result.message_strategy is not None
        assert result.message_strategy.tone is not None
        assert len(result.message_strategy.empathy_points) > 0
        assert len(result.message_strategy.key_points) > 0

    def test_is_valid_model(self):
        """Should be a valid DirectorResponse."""
        result = get_fallback_strategy()
        assert result.strategic_reasoning is not None


class TestFallbackMessage:
    """Tests for executor fallback."""

    def test_spanish_message(self):
        """Should return Spanish message."""
        result = get_fallback_message_spanish()
        assert "Gracias" in result.message.content
        assert "tecnicas" in result.message.content

    def test_english_message(self):
        """Should return English message."""
        result = get_fallback_message_english()
        assert "Thank you" in result.message.content
        assert "technical" in result.message.content

    def test_language_selector_spanish(self):
        """Should return Spanish by default."""
        result = get_fallback_message("spanish")
        assert "Gracias" in result.message.content

    def test_language_selector_english(self):
        """Should return English when requested."""
        result = get_fallback_message("english")
        assert "Thank you" in result.message.content

    def test_language_selector_default(self):
        """Should default to Spanish for unknown language."""
        result = get_fallback_message("french")
        assert "Gracias" in result.message.content

    def test_is_valid_model(self):
        """Should be valid ExecutorResponse."""
        result = get_fallback_message()
        assert result.message.content is not None
        assert result.execution_summary is not None
        assert 0 <= result.agreement_level <= 1.0

    def test_feedback_indicates_fallback(self):
        """Should indicate this was a fallback in feedback."""
        result = get_fallback_message()
        assert "fallback" in result.feedback_for_director.lower() or \
               "unavailable" in result.feedback_for_director.lower()
