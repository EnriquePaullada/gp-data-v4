import pytest
import datetime as dt
import os
from src.models.lead import Lead
from src.models.message import Message, MessageRole
from dotenv import load_dotenv

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

def pytest_configure(config):
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️ WARNING: OPENAI_API_KEY not found. AI tests will be skipped.")
    if not os.getenv("MONGODB_URI"):
        print("\n⚠️ WARNING: MONGODB_URI not found. Repository tests will be skipped.")

    # Register custom markers
    config.addinivalue_line("markers", "mongodb: mark test as requiring MongoDB")


def pytest_collection_modifyitems(config, items):
    """Skip MongoDB tests if MONGODB_URI not configured."""
    if os.getenv("MONGODB_URI"):
        return  # MongoDB available, run all tests

    skip_mongodb = pytest.mark.skip(reason="MONGODB_URI not configured")
    for item in items:
        # Skip all tests in repositories directory
        if "test_repositories" in str(item.fspath) or "repositories" in str(item.fspath):
            item.add_marker(skip_mongodb)