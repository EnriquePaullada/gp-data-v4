import asyncio
import pytest
import datetime as dt
import os
import mongomock_motor 
from src.config import settings
from src.models.lead import Lead
from src.models.message import Message, MessageRole
from dotenv import load_dotenv

from src.repositories.connection import db_manager
from src.utils.cost_tracker import get_cost_tracker

os.environ["ENVIRONMENT"] = "test"
os.environ["MONGODB_DATABASE"] = "gp-data-v4-development"

load_dotenv()


@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Mocks the entire database layer for the test session."""
    # 1. Create a fake MongoDB in memory
    mock_client = mongomock_motor.AsyncMongoMockClient() 
    
    async def mock_connect():
        db_manager._client = mock_client
        db_manager._database = mock_client[settings.mongodb_database]
        return None
        
    db_manager.connect = mock_connect
    
    asyncio.run(mock_connect())

@pytest.fixture(scope="session", autouse=True)
def load_env():
    """Ensure env vars are loaded for the entire session."""
    load_dotenv()

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

@pytest.fixture(autouse=True)
async def clear_database_collections():
    """
    WARRIOR DISCIPLINE: Wipe all data after every test.
    This prevents DuplicateKeyErrors between independent tests.
    """
    
    tracker = get_cost_tracker()
    tracker.hourly_usage.total_cost = 0.0
    tracker.daily_usage.total_cost = 0.0

    yield
    db = db_manager._database
    if db is not None:
        collections = await db.list_collection_names()
        for coll in collections:
            await db[coll].delete_many({})

    