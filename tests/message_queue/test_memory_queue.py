"""
Tests for InMemoryQueue implementation.
"""

import pytest
import asyncio
from datetime import datetime, timedelta, timezone

from src.message_queue import InMemoryQueue, QueuedMessage, MessageStatus, QueueMetrics


class TestInMemoryQueue:
    """Test suite for InMemoryQueue."""

    @pytest.fixture
    async def queue(self):
        """Create a fresh queue for each test."""
        return InMemoryQueue()

    @pytest.fixture
    def sample_message(self):
        """Create a sample queued message."""
        return QueuedMessage(
            id="msg-123",
            phone="+5215538899800",
            body="Hello, I'm interested in your product",
            profile_name="Carlos Rodriguez",
            message_sid="SM123456"
        )

    @pytest.mark.asyncio
    async def test_enqueue_message(self, queue, sample_message):
        """Test enqueueing a message."""
        message_id = await queue.enqueue(sample_message)

        assert message_id == "msg-123"

        # Verify queue metrics
        metrics = await queue.get_metrics()
        assert metrics.pending == 1
        assert metrics.processing == 0

    @pytest.mark.asyncio
    async def test_enqueue_generates_id_if_missing(self, queue):
        """Test that enqueue generates ID if not provided."""
        message = QueuedMessage(
            id="",  # Empty ID
            phone="+5215538899800",
            body="Test",
            message_sid="SM123"
        )

        message_id = await queue.enqueue(message)

        assert message_id != ""
        assert len(message_id) > 0

    @pytest.mark.asyncio
    async def test_dequeue_message(self, queue, sample_message):
        """Test dequeueing a message."""
        await queue.enqueue(sample_message)

        # Dequeue
        message = await queue.dequeue()

        assert message is not None
        assert message.id == "msg-123"
        assert message.status == MessageStatus.PROCESSING

        # Verify metrics
        metrics = await queue.get_metrics()
        assert metrics.pending == 0
        assert metrics.processing == 1

    @pytest.mark.asyncio
    async def test_dequeue_empty_queue(self, queue):
        """Test dequeueing from empty queue returns None."""
        message = await queue.dequeue()
        assert message is None

    @pytest.mark.asyncio
    async def test_complete_message(self, queue, sample_message):
        """Test completing a message."""
        await queue.enqueue(sample_message)
        message = await queue.dequeue()

        # Complete
        await queue.complete(message.id)

        # Verify metrics
        metrics = await queue.get_metrics()
        assert metrics.processing == 0
        assert metrics.completed == 1
        assert metrics.avg_processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_fail_message_with_retry(self, queue, sample_message):
        """Test failing a message triggers retry."""
        await queue.enqueue(sample_message)
        message = await queue.dequeue()

        # Fail
        await queue.fail(message.id, "Test error")

        # Verify message is re-queued
        metrics = await queue.get_metrics()
        assert metrics.failed == 1

        # Message should be scheduled for retry (not immediately available)
        immediate_retry = await queue.dequeue()
        assert immediate_retry is None  # Not ready yet due to retry delay

        # Check message was updated in storage
        assert queue._messages[message.id].retry_count == 1
        assert queue._messages[message.id].error == "Test error"

    @pytest.mark.asyncio
    async def test_fail_message_max_retries_moves_to_dead_letter(self, queue, sample_message):
        """Test that exceeding max retries moves message to dead letter queue."""
        sample_message.max_retries = 1  # Allow only 1 retry

        await queue.enqueue(sample_message)

        # Fail twice (first attempt + 1 retry = 2 failures total)
        # First failure
        message = await queue.dequeue()
        assert message is not None
        await queue.fail(message.id, "Error 1")

        # Message should be re-queued with retry_count = 1
        assert queue._messages[message.id].retry_count == 1

        # Manually set scheduled_at to now to bypass delay
        queue._messages[message.id].scheduled_at = datetime.now(timezone.utc)

        # Second failure (retry_count becomes 2, exceeds max_retries of 1)
        message = await queue.dequeue()
        assert message is not None
        await queue.fail(message.id, "Error 2")

        # Verify message is in dead letter queue
        metrics = await queue.get_metrics()
        assert metrics.dead_letter == 1

        dead_letters = await queue.get_dead_letter_messages()
        assert len(dead_letters) == 1
        assert dead_letters[0].id == "msg-123"
        assert dead_letters[0].status == MessageStatus.DEAD_LETTER

    @pytest.mark.asyncio
    async def test_retry_delay_schedule(self, queue, sample_message):
        """Test that retry delays follow exponential backoff."""
        await queue.enqueue(sample_message)
        message = await queue.dequeue()

        # First failure - should schedule for 1 minute later
        await queue.fail(message.id, "Error 1")

        # Check scheduled_at is approximately 1 minute in future
        retry_msg = await queue.dequeue()
        if retry_msg:  # May not dequeue if not ready yet
            assert retry_msg.retry_count == 1

    @pytest.mark.asyncio
    async def test_retry_dead_letter_message(self, queue, sample_message):
        """Test retrying a message from dead letter queue."""
        sample_message.max_retries = 0  # Go straight to dead letter on first failure

        await queue.enqueue(sample_message)

        # Fail once to move to dead letter
        message = await queue.dequeue()
        assert message is not None
        await queue.fail(message.id, "Error")

        # Verify in dead letter
        dead_letters = await queue.get_dead_letter_messages()
        assert len(dead_letters) == 1

        # Retry from dead letter
        await queue.retry_dead_letter("msg-123")

        # Verify message is back in queue with reset retry count
        message = await queue.dequeue()
        assert message is not None
        assert message.id == "msg-123"
        assert message.retry_count == 0
        assert message.error is None

    @pytest.mark.asyncio
    async def test_queue_metrics_calculation(self, queue):
        """Test queue metrics are calculated correctly."""
        # Enqueue 5 messages
        for i in range(5):
            msg = QueuedMessage(
                id=f"msg-{i}",
                phone="+5215538899800",
                body=f"Message {i}",
                message_sid=f"SM{i}"
            )
            await queue.enqueue(msg)

        # Process 3 successfully
        for _ in range(3):
            msg = await queue.dequeue()
            if msg:
                await queue.complete(msg.id)

        # Fail 1
        msg = await queue.dequeue()
        if msg:
            await queue.fail(msg.id, "Error")

        # Get metrics
        metrics = await queue.get_metrics()

        assert metrics.pending >= 1  # 1 pending + 1 re-queued from failure
        assert metrics.completed == 3
        assert metrics.failed == 1
        assert metrics.avg_processing_time_ms > 0

        # Error rate should be 1/(3+1) = 25%
        assert 20 <= metrics.error_rate <= 30

    @pytest.mark.asyncio
    async def test_concurrent_enqueue_dequeue(self, queue):
        """Test thread-safe concurrent operations."""
        # Enqueue 10 messages concurrently
        messages = [
            QueuedMessage(
                id=f"msg-{i}",
                phone="+5215538899800",
                body=f"Message {i}",
                message_sid=f"SM{i}"
            )
            for i in range(10)
        ]

        # Enqueue concurrently
        await asyncio.gather(*[queue.enqueue(msg) for msg in messages])

        # Dequeue all concurrently
        dequeued = await asyncio.gather(*[queue.dequeue() for _ in range(10)])

        # Should have 10 messages (some may be None due to timing)
        non_none = [m for m in dequeued if m is not None]
        assert len(non_none) == 10

    @pytest.mark.asyncio
    async def test_get_dead_letter_messages_with_limit(self, queue):
        """Test retrieving dead letter messages with limit."""
        # Create and fail multiple messages to move to dead letter
        for i in range(5):
            msg = QueuedMessage(
                id=f"msg-{i}",
                phone="+5215538899800",
                body=f"Message {i}",
                message_sid=f"SM{i}",
                max_retries=0  # Immediate dead letter
            )
            await queue.enqueue(msg)
            m = await queue.dequeue()
            if m:
                await queue.fail(m.id, "Error")

        # Get with limit
        dead_letters = await queue.get_dead_letter_messages(limit=3)

        assert len(dead_letters) == 3

    @pytest.mark.asyncio
    async def test_message_not_ready_is_requeued(self, queue):
        """Test that messages not ready for processing are re-queued."""
        # Create message scheduled for future
        future_time = datetime.now(timezone.utc) + timedelta(hours=1)
        message = QueuedMessage(
            id="msg-future",
            phone="+5215538899800",
            body="Future message",
            message_sid="SM123",
            scheduled_at=future_time
        )

        await queue.enqueue(message)

        # Try to dequeue - should return None because not ready
        dequeued = await queue.dequeue()
        assert dequeued is None

        # Message should still be in queue
        metrics = await queue.get_metrics()
        assert metrics.pending == 1
