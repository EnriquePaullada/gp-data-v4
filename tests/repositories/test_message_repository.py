"""
Message Repository Tests
Comprehensive tests for Message persistence and conversation queries.
"""
import pytest
import datetime as dt
from typing import List

from src.repositories import db_manager, MessageRepository
from src.models.message import Message, MessageRole


pytestmark = pytest.mark.asyncio


@pytest.fixture(scope="function")
async def message_repo():
    """Provide a clean MessageRepository instance with connected database."""
    await db_manager.connect()
    await db_manager.create_indexes()
    db = db_manager.database

    # Clean up test data before each test
    await db.messages.delete_many({"lead_id": {"$regex": "^\\+test"}})

    repo = MessageRepository(db)
    yield repo

    # Cleanup after test
    await db.messages.delete_many({"lead_id": {"$regex": "^\\+test"}})


@pytest.fixture
def sample_messages() -> List[Message]:
    """Provide a sample conversation for testing."""
    now = dt.datetime.now(dt.UTC)
    lead_id = "+test5215538899800"

    return [
        Message(
            lead_id=lead_id,
            role=MessageRole.LEAD,
            content="Hello, I'm interested",
            timestamp=now - dt.timedelta(minutes=10),
            tokens=5
        ),
        Message(
            lead_id=lead_id,
            role=MessageRole.ASSISTANT,
            content="Great! How can I help?",
            timestamp=now - dt.timedelta(minutes=9),
            tokens=8
        ),
        Message(
            lead_id=lead_id,
            role=MessageRole.LEAD,
            content="What's the pricing?",
            timestamp=now - dt.timedelta(minutes=5),
            tokens=4
        ),
    ]


class TestMessageRepositoryBasicCRUD:
    """Test basic CRUD operations."""

    async def test_create_message(self, message_repo: MessageRepository):
        """Should create a message and return it with _id populated."""
        message = Message(
            lead_id="+test1234567890",
            role=MessageRole.LEAD,
            content="Test message",
            tokens=3
        )

        created_message = await message_repo.create(message)

        assert created_message.id is not None
        assert created_message.lead_id == message.lead_id
        assert created_message.content == message.content
        assert created_message.created_at is not None

    async def test_bulk_create_messages(
        self,
        message_repo: MessageRepository,
        sample_messages: List[Message]
    ):
        """Should create multiple messages in single operation."""
        created_messages = await message_repo.bulk_create(sample_messages)

        assert len(created_messages) == 3
        assert all(msg.id is not None for msg in created_messages)

    async def test_find_by_id(self, message_repo: MessageRepository):
        """Should retrieve a message by MongoDB ObjectId."""
        message = Message(
            lead_id="+test1234567890",
            role=MessageRole.ASSISTANT,
            content="Test response",
            tokens=5
        )

        created = await message_repo.create(message)
        found = await message_repo.find_by_id(created.id)

        assert found is not None
        assert found.id == created.id
        assert found.content == created.content


class TestMessageRepositoryConversationQueries:
    """Test conversation history retrieval."""

    async def test_get_conversation_history(
        self,
        message_repo: MessageRepository,
        sample_messages: List[Message]
    ):
        """Should retrieve full conversation history for a lead."""
        await message_repo.bulk_create(sample_messages)

        history = await message_repo.get_conversation_history(
            lead_id=sample_messages[0].lead_id,
            limit=100
        )

        assert len(history) == 3
        # Should be chronological order
        assert history[0].content == "Hello, I'm interested"
        assert history[1].role == MessageRole.ASSISTANT
        assert history[2].content == "What's the pricing?"

    async def test_get_conversation_history_pagination(
        self,
        message_repo: MessageRepository,
        sample_messages: List[Message]
    ):
        """Should support pagination with before parameter."""
        await message_repo.bulk_create(sample_messages)

        # Get messages before the second message timestamp
        cutoff = sample_messages[1].timestamp

        history = await message_repo.get_conversation_history(
            lead_id=sample_messages[0].lead_id,
            before=cutoff
        )

        # Should only get first message
        assert len(history) == 1
        assert history[0].content == "Hello, I'm interested"

    async def test_get_conversation_history_empty(self, message_repo: MessageRepository):
        """Should return empty list for lead with no messages."""
        history = await message_repo.get_conversation_history(
            lead_id="+test9999999999"
        )

        assert history == []

    async def test_get_recent_messages(
        self,
        message_repo: MessageRepository,
        sample_messages: List[Message]
    ):
        """Should retrieve most recent N messages in chronological order."""
        await message_repo.bulk_create(sample_messages)

        recent = await message_repo.get_recent_messages(
            lead_id=sample_messages[0].lead_id,
            limit=2
        )

        assert len(recent) == 2
        # Should be chronological (oldest first)
        assert recent[0].role == MessageRole.ASSISTANT
        assert recent[1].content == "What's the pricing?"


class TestMessageRepositorySaveMessages:
    """Test save_messages bulk operation (orchestrator integration)."""

    async def test_save_messages_creates_new(
        self,
        message_repo: MessageRepository,
        sample_messages: List[Message]
    ):
        """Should persist new messages without IDs."""
        saved = await message_repo.save_messages(sample_messages)

        assert len(saved) == 3
        assert all(msg.id is not None for msg in saved)

    async def test_save_messages_skips_existing(
        self,
        message_repo: MessageRepository,
        sample_messages: List[Message]
    ):
        """Should skip messages that already have IDs."""
        # Create messages first
        created = await message_repo.bulk_create(sample_messages)

        # Try to save again (IDs are populated)
        saved = await message_repo.save_messages(created)

        # Should return same messages without creating duplicates
        assert len(saved) == 3

        # Verify no duplicates in database
        count = await message_repo.count({"lead_id": sample_messages[0].lead_id})
        assert count == 3

    async def test_save_messages_empty_list(self, message_repo: MessageRepository):
        """Should handle empty message list gracefully."""
        saved = await message_repo.save_messages([])

        assert saved == []


class TestMessageRepositoryFilterQueries:
    """Test message filtering and analytics queries."""

    async def test_get_messages_by_role(
        self,
        message_repo: MessageRepository,
        sample_messages: List[Message]
    ):
        """Should filter messages by role."""
        await message_repo.bulk_create(sample_messages)

        lead_messages = await message_repo.get_messages_by_role(
            lead_id=sample_messages[0].lead_id,
            role=MessageRole.LEAD
        )

        assert len(lead_messages) == 2
        assert all(msg.role == MessageRole.LEAD for msg in lead_messages)

    async def test_get_messages_in_timerange(
        self,
        message_repo: MessageRepository,
        sample_messages: List[Message]
    ):
        """Should retrieve messages within a time window."""
        await message_repo.bulk_create(sample_messages)

        now = dt.datetime.now(dt.UTC)
        start = now - dt.timedelta(minutes=11)
        end = now - dt.timedelta(minutes=8)

        messages = await message_repo.get_messages_in_timerange(
            lead_id=sample_messages[0].lead_id,
            start_time=start,
            end_time=end
        )

        # Should get first 2 messages
        assert len(messages) == 2

    async def test_count_messages_for_lead(
        self,
        message_repo: MessageRepository,
        sample_messages: List[Message]
    ):
        """Should count total messages for a lead."""
        await message_repo.bulk_create(sample_messages)

        count = await message_repo.count_messages_for_lead(
            lead_id=sample_messages[0].lead_id
        )

        assert count == 3

    async def test_get_total_tokens_for_lead(
        self,
        message_repo: MessageRepository,
        sample_messages: List[Message]
    ):
        """Should sum total tokens used in conversation."""
        await message_repo.bulk_create(sample_messages)

        total_tokens = await message_repo.get_total_tokens_for_lead(
            lead_id=sample_messages[0].lead_id
        )

        # 5 + 8 + 4 = 17
        assert total_tokens == 17

    async def test_get_total_tokens_for_lead_no_messages(
        self,
        message_repo: MessageRepository
    ):
        """Should return 0 tokens for lead with no messages."""
        total = await message_repo.get_total_tokens_for_lead("+test9999999999")

        assert total == 0


class TestMessageRepositoryAnalytics:
    """Test business intelligence and analytics queries."""

    async def test_get_average_response_time(
        self,
        message_repo: MessageRepository
    ):
        """Should calculate average response time between lead and assistant."""
        now = dt.datetime.now(dt.UTC)
        lead_id = "+test1234567890"

        messages = [
            Message(
                lead_id=lead_id,
                role=MessageRole.LEAD,
                content="Question 1",
                timestamp=now - dt.timedelta(minutes=10),
                tokens=3
            ),
            Message(
                lead_id=lead_id,
                role=MessageRole.ASSISTANT,
                content="Answer 1",
                timestamp=now - dt.timedelta(minutes=9),  # 1 min response
                tokens=5
            ),
            Message(
                lead_id=lead_id,
                role=MessageRole.LEAD,
                content="Question 2",
                timestamp=now - dt.timedelta(minutes=5),
                tokens=3
            ),
            Message(
                lead_id=lead_id,
                role=MessageRole.ASSISTANT,
                content="Answer 2",
                timestamp=now - dt.timedelta(minutes=2),  # 3 min response
                tokens=5
            ),
        ]

        await message_repo.bulk_create(messages)

        avg_time = await message_repo.get_average_response_time(
            lead_id=lead_id,
            hours=24
        )

        # Average of 60s and 180s = 120s
        assert avg_time is not None
        assert 115 <= avg_time <= 125  # Allow small variance

    async def test_get_average_response_time_insufficient_data(
        self,
        message_repo: MessageRepository
    ):
        """Should return None if insufficient data."""
        avg_time = await message_repo.get_average_response_time(
            lead_id="+test9999999999",
            hours=24
        )

        assert avg_time is None

    async def test_get_average_response_time_no_responses(
        self,
        message_repo: MessageRepository
    ):
        """Should return None if no assistant responses."""
        message = Message(
            lead_id="+test1234567890",
            role=MessageRole.LEAD,
            content="Only lead message",
            tokens=5
        )

        await message_repo.create(message)

        avg_time = await message_repo.get_average_response_time(
            lead_id=message.lead_id,
            hours=24
        )

        assert avg_time is None


class TestMessageRepositoryDeletion:
    """Test message deletion (GDPR compliance)."""

    async def test_delete_messages_for_lead(
        self,
        message_repo: MessageRepository,
        sample_messages: List[Message]
    ):
        """Should delete all messages for a specific lead."""
        await message_repo.bulk_create(sample_messages)

        deleted_count = await message_repo.delete_messages_for_lead(
            lead_id=sample_messages[0].lead_id
        )

        assert deleted_count == 3

        # Verify deletion
        count = await message_repo.count_messages_for_lead(
            lead_id=sample_messages[0].lead_id
        )
        assert count == 0

    async def test_delete_messages_for_lead_nonexistent(
        self,
        message_repo: MessageRepository
    ):
        """Should return 0 when deleting messages for non-existent lead."""
        deleted_count = await message_repo.delete_messages_for_lead(
            "+test9999999999"
        )

        assert deleted_count == 0


@pytest.fixture(scope="module", autouse=True)
async def cleanup_database():
    """Clean up test database after all tests."""
    yield
    await db_manager.connect()
    await db_manager.database.messages.delete_many({"lead_id": {"$regex": "^\\+test"}})
    await db_manager.disconnect()
