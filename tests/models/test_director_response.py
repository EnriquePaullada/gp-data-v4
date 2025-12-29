import pytest
from pydantic import ValidationError
from src.models.director_response import DirectorResponse, StrategicAction
from src.models.intelligence import BANTDimension


def test_director_response_valid():
    """Verifies that a valid strategy packet can be created."""
    data = {
        "action": StrategicAction.QUALIFY,
        "strategic_reasoning": "Need budget.",
        "focus_dimension": BANTDimension.BUDGET,
        "message_strategy": {
            "tone": "consultative",
            "language": "english",
            "empathy_points": ["Understand scaling"],
            "key_points": ["ROI of $99 plan"], # FIX: These live here now
            "conversational_goal": "Get budget"
        }
    }
    response = DirectorResponse(**data)
    assert response.action == StrategicAction.QUALIFY
    assert len(response.message_strategy.key_points) == 1

def test_director_response_missing_fields():
    """Ensures the model fails if the message strategy is missing."""
    with pytest.raises(ValidationError):
        # Action is provided, but message_strategy (required) is not
        DirectorResponse(
            action=StrategicAction.NURTURE,
            strategic_reasoning="Test",
            key_points_to_address=[]
        )

def test_director_response_invalid_action():
    """Ensures Pydantic catches invalid Enum values."""
    with pytest.raises(ValidationError):
        DirectorResponse(
            action="invalid_action_name", # Not in Enum
            strategic_reasoning="Reasoning...",
            key_points_to_address=[],
            message_strategy={"tone": "warm", "language": "english", "empathy_points": [], "key_points": [], "conversational_goal": ""}
        )