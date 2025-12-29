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
    assert "?" in response.message.content # Alena must ask the conversational_goal
    assert "burnout" in response.message.content.lower() # Alena must use empathy_points
    
    print(f"\nAlena's Message: {response.message.content}")