import pytest
import os
from src.agents.director_agent import DirectorService
from src.models.classifier_response import ClassifierResponse, Intent
from src.models.director_response import StrategicAction
from src.models.message import Message, MessageRole


@pytest.fixture
def mock_classification_factory():
    """Helper to create valid classification objects for testing."""
    def _create(intent: Intent):
        return ClassifierResponse(
            intent=intent,
            intent_confidence=1.0,
            topic="general",
            topic_confidence=1.0, 
            urgency="low",
            urgency_confidence=1.0,
            language="english",
            sentiment="neutral",
            engagement_level="medium",
            requires_human_escalation=False,
            reasoning="Test reasoning",
            new_signals=[]
        )
    return _create

@pytest.mark.asyncio
async def test_director_hard_gate_logic(mock_lead, mock_classification_factory):
    """Verifies that READY_TO_BUY triggers a deterministic CLOSE."""
    service = DirectorService()
    
    # 1. Setup high-intent mock
    mock_classification = mock_classification_factory(Intent.READY_TO_BUY)
    mock_classification.language = "spanish"
    
    # 2. Execute
    response = await service.decide_next_move(mock_lead, mock_classification)
    
    # 3. Assertions
    assert response.action == StrategicAction.CLOSE
    assert response.message_strategy.language == "spanish"
    # Ensure the 'Hard-gate' keyword is present as defined in the service
    assert "Hard-gate" in response.strategic_reasoning
    assert "Bypassing" in response.strategic_reasoning


@pytest.mark.asyncio
async def test_director_llm_reasoning(mock_lead, mock_classification_factory):
    """Verifies the LLM integration logic."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("No API Key found")

    service = DirectorService()
    
    mock_classification = mock_classification_factory(Intent.FOLLOWUP)
    
    response = await service.decide_next_move(mock_lead, mock_classification)
    
    assert response.action is not None
    assert len(response.message_strategy.empathy_points) > 0

@pytest.mark.asyncio
async def test_director_context_awareness(mock_lead, mock_classification_factory): # Added factory
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("No API Key found")

    service = DirectorService()

    # 1. Give the lead some history
    msg = Message(lead_id=mock_lead.lead_id, role=MessageRole.LEAD, content="Our team has 40 people.")
    mock_lead.add_message(msg)

    # 2. Use the FACTORY to ensure the object is valid
    mock_classification = mock_classification_factory(Intent.FEATURE)
    mock_classification.reasoning = "User mentioned team size."

    response = await service.decide_next_move(mock_lead, mock_classification)

    assert response.action is not None