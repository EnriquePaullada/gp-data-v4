import pytest
import os
from src.agents.executor_agent import ExecutorService
from src.models.director_response import DirectorResponse, StrategicAction, MessageStrategy

@pytest.mark.asyncio
async def test_alena_compliance_and_tone(mock_lead):
    """
    Verifies Alena follows strategy and maintains compliance.
    """
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("No API Key found")

    service = ExecutorService()
    strategy = DirectorResponse(
        action=StrategicAction.QUALIFY,
        strategic_reasoning="Lead mentioned burnout; need to confirm team size.",
        message_strategy=MessageStrategy(
            tone="empathetic",
            language="english",
            empathy_points=["Understand how draining manual follow-up is"],
            key_points=["Workshop fixes process burnout"],
            conversational_goal="Confirm if they have 20+ reps"
        )
    )

    response = await service.craft_message(mock_lead, strategy)

    assert response.agreement_level >= 0.8
    # Alena must ask/inquire about the conversational_goal (question mark or question-like phrasing)
    content_lower = response.message.content.lower()
    asks_question = (
        "?" in response.message.content or
        "love to know" in content_lower or
        "let me know" in content_lower or
        "could you" in content_lower or
        "would you" in content_lower
    )
    assert asks_question, f"Expected question or inquiry in: {response.message.content}"
    # Alena must use empathy points - check for any empathy-related terms
    empathy_words = ["burnout", "draining", "exhausting", "manual follow-up", "process"]
    assert any(word in content_lower for word in empathy_words), f"No empathy words found in: {response.message.content}"
    
    print(f"\nAlena's Message: {response.message.content}")