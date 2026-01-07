"""
Tests for QueueWorker.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock

from src.message_queue import InMemoryQueue, QueueWorker, QueuedMessage


class TestQueueWorker:
    """Test suite for QueueWorker."""

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
            body="Test message",
            message_sid="SM123"
        )

    @pytest.mark.asyncio
    async def test_worker_processes_message(self, queue, sample_message):
        """Test that worker processes enqueued messages."""
        # Track if handler was called
        handler_called = asyncio.Event()
        processed_message = None

        async def test_handler(message: QueuedMessage):
            nonlocal processed_message
            processed_message = message
            handler_called.set()

        # Create worker
        worker = QueueWorker(
            queue=queue,
            handler=test_handler,
            poll_interval=0.1
        )

        # Enqueue message
        await queue.enqueue(sample_message)

        # Start worker in background
        worker_task = asyncio.create_task(worker.start())

        try:
            # Wait for handler to be called
            await asyncio.wait_for(handler_called.wait(), timeout=2.0)

            # Verify message was processed
            assert processed_message is not None
            assert processed_message.id == "msg-123"
            assert processed_message.phone == "+5215538899800"

            # Verify message marked as completed
            metrics = await queue.get_metrics()
            assert metrics.completed == 1

        finally:
            await worker.stop()
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_worker_handles_handler_failure(self, queue, sample_message):
        """Test that worker handles handler failures and triggers retry."""
        # Handler that always fails
        async def failing_handler(message: QueuedMessage):
            raise ValueError("Intentional failure")

        worker = QueueWorker(
            queue=queue,
            handler=failing_handler,
            poll_interval=0.1
        )

        # Enqueue message
        await queue.enqueue(sample_message)

        # Start worker
        worker_task = asyncio.create_task(worker.start())

        try:
            # Wait a bit for processing
            await asyncio.sleep(0.5)

            # Message should have failed and been re-queued
            metrics = await queue.get_metrics()
            assert metrics.failed == 1

        finally:
            await worker.stop()
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_worker_respects_max_concurrent(self, queue):
        """Test that worker respects max_concurrent limit."""
        processing_count = 0
        max_seen = 0
        lock = asyncio.Lock()

        async def slow_handler(message: QueuedMessage):
            nonlocal processing_count, max_seen

            async with lock:
                processing_count += 1
                max_seen = max(max_seen, processing_count)

            # Simulate slow processing
            await asyncio.sleep(0.2)

            async with lock:
                processing_count -= 1

        # Create worker with max_concurrent=2
        worker = QueueWorker(
            queue=queue,
            handler=slow_handler,
            max_concurrent=2,
            poll_interval=0.01
        )

        # Enqueue 5 messages
        for i in range(5):
            msg = QueuedMessage(
                id=f"msg-{i}",
                phone="+5215538899800",
                body=f"Message {i}",
                message_sid=f"SM{i}"
            )
            await queue.enqueue(msg)

        # Start worker
        worker_task = asyncio.create_task(worker.start())

        try:
            # Wait for all messages to process
            await asyncio.sleep(1.0)

            # Max concurrent should not exceed 2
            assert max_seen <= 2

        finally:
            await worker.stop()
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_worker_graceful_shutdown(self, queue):
        """Test that worker shuts down gracefully."""
        shutdown_complete = asyncio.Event()

        async def handler(message: QueuedMessage):
            # Simulate some processing
            await asyncio.sleep(0.1)

        worker = QueueWorker(
            queue=queue,
            handler=handler,
            poll_interval=0.1
        )

        # Enqueue a message
        msg = QueuedMessage(
            id="msg-1",
            phone="+5215538899800",
            body="Test",
            message_sid="SM1"
        )
        await queue.enqueue(msg)

        # Start worker
        worker_task = asyncio.create_task(worker.start())

        # Let it process for a bit
        await asyncio.sleep(0.2)

        # Stop worker
        await worker.stop()

        # Verify shutdown completed
        assert worker._running is False

        # Cleanup
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_worker_processes_multiple_messages(self, queue):
        """Test that worker processes multiple messages in sequence."""
        processed_ids = []
        lock = asyncio.Lock()

        async def tracking_handler(message: QueuedMessage):
            async with lock:
                processed_ids.append(message.id)

        worker = QueueWorker(
            queue=queue,
            handler=tracking_handler,
            poll_interval=0.05
        )

        # Enqueue 5 messages
        for i in range(5):
            msg = QueuedMessage(
                id=f"msg-{i}",
                phone="+5215538899800",
                body=f"Message {i}",
                message_sid=f"SM{i}"
            )
            await queue.enqueue(msg)

        # Start worker
        worker_task = asyncio.create_task(worker.start())

        try:
            # Wait for processing
            await asyncio.sleep(1.0)

            # All 5 messages should be processed
            assert len(processed_ids) == 5
            assert set(processed_ids) == {f"msg-{i}" for i in range(5)}

            # Verify metrics
            metrics = await queue.get_metrics()
            assert metrics.completed == 5

        finally:
            await worker.stop()
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_worker_idle_when_no_messages(self, queue):
        """Test that worker polls correctly when queue is empty."""
        handler_call_count = 0

        async def counting_handler(message: QueuedMessage):
            nonlocal handler_call_count
            handler_call_count += 1

        worker = QueueWorker(
            queue=queue,
            handler=counting_handler,
            poll_interval=0.1
        )

        # Start worker with empty queue
        worker_task = asyncio.create_task(worker.start())

        try:
            # Let it run for a bit
            await asyncio.sleep(0.5)

            # Handler should not have been called
            assert handler_call_count == 0

        finally:
            await worker.stop()
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_worker_with_mock_handler(self, queue, sample_message):
        """Test worker with mocked handler."""
        mock_handler = AsyncMock()

        worker = QueueWorker(
            queue=queue,
            handler=mock_handler,
            poll_interval=0.1
        )

        await queue.enqueue(sample_message)

        worker_task = asyncio.create_task(worker.start())

        try:
            # Wait for processing
            await asyncio.sleep(0.5)

            # Verify handler was called
            mock_handler.assert_called_once()

            # Verify called with correct message
            call_args = mock_handler.call_args[0]
            assert call_args[0].id == "msg-123"

        finally:
            await worker.stop()
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass
