import pytest
import datetime as dt
from src.models.lead import Lead
from src.models.message import Message, MessageRole

@pytest.fixture
def mock_lead():
    """Returns a clean Lead object for testing."""
    return Lead(lead_id="+1234567890", full_name="Test User")

@pytest.fixture
def mock_lead_message(mock_lead):
    """Returns a valid lead message."""
    return Message(
        lead_id=mock_lead.lead_id,
        role=MessageRole.LEAD,
        content="I am interested in pricing for 10 users.",
        timestamp=dt.datetime.now(dt.UTC)
    )