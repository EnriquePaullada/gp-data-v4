import pytest
import os
from src.agents.director_agent import DirectorService
from src.models.intelligence import IntelligenceSignal, BANTDimension, ConfidenceScore

@pytest.mark.asyncio
async def test_director_llm_reasoning(mock_lead):
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("No API Key")

    service = DirectorService()
    need_signal = IntelligenceSignal(
        dimension=BANTDimension.NEED,
        extracted_value="Automate WhatsApp followups",
        confidence=ConfidenceScore(value=1.0, reasoning="Directly stated"),
        source_message_id="test_msg_1",
        raw_evidence="I want to automate my team"
    )
    mock_lead.add_signal(need_signal)
    
    response = await service.decide_next_move(mock_lead, "followup_response")
    
    assert response.action is not None
    # Verify the nested strategy exists
    assert len(response.message_strategy.key_points) > 0