"""
Fallback Responses for LLM Degradation

Predefined safe responses when OpenAI is unavailable.
These keep conversations alive without making assumptions.
"""

from src.models.classifier_response import (
    ClassifierResponse,
    Intent,
    UrgencyLevel,
)
from src.models.director_response import (
    DirectorResponse,
    StrategicAction,
    MessageStrategy,
)
from src.models.executor_response import (
    ExecutorResponse,
    OutboundMessage,
)
from src.models.intelligence import Sentiment, BANTDimension


def get_fallback_classification() -> ClassifierResponse:
    """
    Safe classification when Classifier agent fails.

    Returns unclear/neutral classification that won't
    trigger aggressive sales actions.
    """
    return ClassifierResponse(
        intent=Intent.UNCLEAR,
        intent_confidence=0.0,
        topic="Unable to classify - service degraded",
        topic_confidence=0.0,
        urgency=UrgencyLevel.UNCLEAR,
        urgency_confidence=0.0,
        language="spanish",  # Default to Spanish for Mexico market
        sentiment=Sentiment.NEUTRAL,
        engagement_level="medium",
        requires_human_escalation=True,  # Flag for human review
        reasoning="Classification unavailable due to service degradation. Flagged for human review.",
        new_signals=[],
    )


def get_fallback_strategy() -> DirectorResponse:
    """
    Safe strategy when Director agent fails.

    Returns a helpful, non-pushy nurture action
    that keeps the conversation warm.
    """
    return DirectorResponse(
        action=StrategicAction.HELP,
        strategic_reasoning="Strategy engine unavailable. Defaulting to helpful response.",
        message_strategy=MessageStrategy(
            tone="warm and helpful",
            language="spanish",
            empathy_points=["Acknowledge their message", "Show we're here to help"],
            key_points=["We received their message", "We'll follow up shortly"],
            conversational_goal="Maintain rapport while service recovers",
        ),
        focus_dimension=BANTDimension.NEED,
        suggested_stage_transition=None,
        scheduling_instruction=None,
    )


def get_fallback_message_spanish() -> ExecutorResponse:
    """
    Safe response message in Spanish when Executor agent fails.

    Polite acknowledgment that doesn't promise anything specific.
    """
    return ExecutorResponse(
        message=OutboundMessage(
            content="Gracias por tu mensaje. En este momento estamos experimentando "
                    "algunas dificultades tecnicas. Te respondere en breve.",
            persona_reasoning="Service degradation fallback - maintains warmth while buying time",
        ),
        agreement_level=0.5,
        feedback_for_director="Executor unavailable - used fallback response",
        execution_summary="Fallback: Technical difficulties message (Spanish)",
    )


def get_fallback_message_english() -> ExecutorResponse:
    """
    Safe response message in English when Executor agent fails.

    Polite acknowledgment that doesn't promise anything specific.
    """
    return ExecutorResponse(
        message=OutboundMessage(
            content="Thank you for your message. We're experiencing some technical "
                    "difficulties at the moment. I'll get back to you shortly.",
            persona_reasoning="Service degradation fallback - maintains warmth while buying time",
        ),
        agreement_level=0.5,
        feedback_for_director="Executor unavailable - used fallback response",
        execution_summary="Fallback: Technical difficulties message (English)",
    )


def get_fallback_message(language: str = "spanish") -> ExecutorResponse:
    """
    Get appropriate fallback message based on language.

    Args:
        language: "spanish" or "english"

    Returns:
        ExecutorResponse with appropriate language
    """
    if language == "english":
        return get_fallback_message_english()
    return get_fallback_message_spanish()
