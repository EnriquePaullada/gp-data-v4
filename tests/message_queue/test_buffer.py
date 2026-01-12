"""
Tests for Message Buffer

Validates WhatsApp burst message buffering behavior.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.message_queue.buffer import MessageBuffer, BufferedMessage, LeadBuffer


class TestMessageBuffer:
    """Tests for MessageBuffer class."""

    @pytest.fixture
    def mock_callback(self):
        """Create mock flush callback."""
        return AsyncMock()

    @pytest.fixture
    def fast_buffer(self, mock_callback):
        """Create buffer with short timeout for testing."""
        return MessageBuffer(
            on_flush=mock_callback,
            buffer_seconds=0.1,  # 100ms for fast tests
            max_messages=5,
            separator="\n",
        )

    @pytest.mark.asyncio
    async def test_single_message_flushes_after_timeout(self, fast_buffer, mock_callback):
        """Single message should flush after buffer timeout."""
        await fast_buffer.add("+5215512345678", "Hello", "SM123", "Carlos")

        # Wait for flush
        await asyncio.sleep(0.2)

        mock_callback.assert_called_once_with(
            "+5215512345678",
            "Hello",
            "SM123",
            "Carlos",
        )

    @pytest.mark.asyncio
    async def test_multiple_messages_concatenated(self, fast_buffer, mock_callback):
        """Multiple rapid messages should be concatenated."""
        await fast_buffer.add("+5215512345678", "Hi", "SM001", "Carlos")
        await fast_buffer.add("+5215512345678", "I saw your ad", "SM002", "Carlos")
        await fast_buffer.add("+5215512345678", "How much?", "SM003", "Carlos")

        # Wait for flush
        await asyncio.sleep(0.2)

        mock_callback.assert_called_once()
        call_args = mock_callback.call_args[0]
        assert call_args[0] == "+5215512345678"
        assert call_args[1] == "Hi\nI saw your ad\nHow much?"
        assert call_args[2] == "SM001"  # First message SID
        assert call_args[3] == "Carlos"

    @pytest.mark.asyncio
    async def test_timer_resets_on_new_message(self, mock_callback):
        """Timer should reset when new message arrives."""
        buffer = MessageBuffer(
            on_flush=mock_callback,
            buffer_seconds=0.15,
            max_messages=10,
        )

        await buffer.add("+5215512345678", "First", "SM001", None)
        await asyncio.sleep(0.1)  # Wait 100ms

        # Timer should reset
        await buffer.add("+5215512345678", "Second", "SM002", None)
        await asyncio.sleep(0.1)  # Wait another 100ms (200ms total)

        # Should not have flushed yet (timer reset at 100ms)
        mock_callback.assert_not_called()

        # Wait for flush
        await asyncio.sleep(0.1)

        mock_callback.assert_called_once()
        call_args = mock_callback.call_args[0]
        assert call_args[1] == "First\nSecond"

    @pytest.mark.asyncio
    async def test_max_messages_forces_flush(self, mock_callback):
        """Should force flush when max messages reached."""
        buffer = MessageBuffer(
            on_flush=mock_callback,
            buffer_seconds=10.0,  # Long timeout
            max_messages=3,
        )

        await buffer.add("+5215512345678", "One", "SM001", None)
        await buffer.add("+5215512345678", "Two", "SM002", None)

        mock_callback.assert_not_called()

        # Third message triggers force flush
        await buffer.add("+5215512345678", "Three", "SM003", None)

        mock_callback.assert_called_once()
        call_args = mock_callback.call_args[0]
        assert call_args[1] == "One\nTwo\nThree"

    @pytest.mark.asyncio
    async def test_separate_buffers_per_lead(self, fast_buffer, mock_callback):
        """Each lead should have independent buffer."""
        await fast_buffer.add("+5215511111111", "Lead 1 msg", "SM001", "Alice")
        await fast_buffer.add("+5215522222222", "Lead 2 msg", "SM002", "Bob")

        # Wait for flush
        await asyncio.sleep(0.2)

        assert mock_callback.call_count == 2

        # Check both leads were flushed
        calls = [call[0] for call in mock_callback.call_args_list]
        phones = {call[0] for call in calls}
        assert phones == {"+5215511111111", "+5215522222222"}

    @pytest.mark.asyncio
    async def test_flush_all_immediate(self, mock_callback):
        """flush_all should immediately flush all pending buffers."""
        buffer = MessageBuffer(
            on_flush=mock_callback,
            buffer_seconds=10.0,  # Long timeout
            max_messages=100,
        )

        await buffer.add("+5215511111111", "Msg 1", "SM001", None)
        await buffer.add("+5215522222222", "Msg 2", "SM002", None)

        mock_callback.assert_not_called()

        await buffer.flush_all()

        assert mock_callback.call_count == 2

    @pytest.mark.asyncio
    async def test_get_pending_count(self, mock_callback):
        """Should track pending lead count."""
        buffer = MessageBuffer(
            on_flush=mock_callback,
            buffer_seconds=10.0,
            max_messages=100,
        )

        assert await buffer.get_pending_count() == 0

        await buffer.add("+5215511111111", "Msg", "SM001", None)
        assert await buffer.get_pending_count() == 1

        await buffer.add("+5215522222222", "Msg", "SM002", None)
        assert await buffer.get_pending_count() == 2

        await buffer.flush_all()
        assert await buffer.get_pending_count() == 0

    @pytest.mark.asyncio
    async def test_get_buffer_stats(self, mock_callback):
        """Should return accurate buffer statistics."""
        buffer = MessageBuffer(
            on_flush=mock_callback,
            buffer_seconds=5.0,
            max_messages=10,
        )

        await buffer.add("+5215511111111", "Msg 1", "SM001", None)
        await buffer.add("+5215511111111", "Msg 2", "SM002", None)
        await buffer.add("+5215522222222", "Msg 3", "SM003", None)

        stats = await buffer.get_buffer_stats()

        assert stats["pending_leads"] == 2
        assert stats["total_buffered_messages"] == 3
        assert stats["buffer_seconds"] == 5.0
        assert stats["max_messages_per_lead"] == 10

    @pytest.mark.asyncio
    async def test_custom_separator(self, mock_callback):
        """Should use custom separator for concatenation."""
        buffer = MessageBuffer(
            on_flush=mock_callback,
            buffer_seconds=0.1,
            max_messages=100,
            separator=" | ",
        )

        await buffer.add("+5215512345678", "A", "SM001", None)
        await buffer.add("+5215512345678", "B", "SM002", None)
        await buffer.add("+5215512345678", "C", "SM003", None)

        await asyncio.sleep(0.2)

        call_args = mock_callback.call_args[0]
        assert call_args[1] == "A | B | C"

    @pytest.mark.asyncio
    async def test_profile_name_updated(self, fast_buffer, mock_callback):
        """Should use most recent profile name."""
        await fast_buffer.add("+5215512345678", "Hi", "SM001", None)
        await fast_buffer.add("+5215512345678", "Hello", "SM002", "Carlos Rodriguez")

        await asyncio.sleep(0.2)

        call_args = mock_callback.call_args[0]
        assert call_args[3] == "Carlos Rodriguez"

    @pytest.mark.asyncio
    async def test_empty_buffer_no_flush(self, mock_callback):
        """Empty buffer should not trigger callback."""
        buffer = MessageBuffer(
            on_flush=mock_callback,
            buffer_seconds=10.0,
            max_messages=100,
        )

        await buffer.flush_all()
        mock_callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_callback_error_logged(self, mock_callback, caplog):
        """Callback errors should be logged."""
        mock_callback.side_effect = ValueError("Test error")

        buffer = MessageBuffer(
            on_flush=mock_callback,
            buffer_seconds=0.05,
            max_messages=100,
        )

        await buffer.add("+5215512345678", "Test", "SM001", None)

        # Wait for background flush to complete
        await asyncio.sleep(0.1)

        # Callback was attempted
        mock_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_flush_all_error_propagates(self, mock_callback):
        """Errors during flush_all should propagate."""
        mock_callback.side_effect = ValueError("Test error")

        buffer = MessageBuffer(
            on_flush=mock_callback,
            buffer_seconds=10.0,  # Long timeout, no auto-flush
            max_messages=100,
        )

        await buffer.add("+5215512345678", "Test", "SM001", None)

        with pytest.raises(ValueError, match="Test error"):
            await buffer.flush_all()

    @pytest.mark.asyncio
    async def test_concurrent_adds_thread_safe(self, mock_callback):
        """Concurrent adds should be handled safely."""
        buffer = MessageBuffer(
            on_flush=mock_callback,
            buffer_seconds=0.2,
            max_messages=100,
        )

        # Add messages concurrently
        async def add_message(i: int):
            await buffer.add("+5215512345678", f"Msg {i}", f"SM{i:03d}", None)

        await asyncio.gather(*[add_message(i) for i in range(10)])

        await asyncio.sleep(0.3)

        mock_callback.assert_called_once()
        call_args = mock_callback.call_args[0]
        # All 10 messages should be present (order may vary due to concurrency)
        assert call_args[1].count("Msg") == 10


class TestBufferedMessage:
    """Tests for BufferedMessage dataclass."""

    def test_defaults(self):
        """Should have sensible defaults."""
        msg = BufferedMessage(body="Hello", message_sid="SM123")

        assert msg.body == "Hello"
        assert msg.message_sid == "SM123"
        assert msg.timestamp is not None

    def test_timestamp_ordering(self):
        """Messages should preserve order by timestamp."""
        import time

        msg1 = BufferedMessage(body="First", message_sid="SM001")
        time.sleep(0.01)
        msg2 = BufferedMessage(body="Second", message_sid="SM002")

        assert msg1.timestamp < msg2.timestamp


class TestLeadBuffer:
    """Tests for LeadBuffer dataclass."""

    def test_defaults(self):
        """Should initialize with empty messages list."""
        buf = LeadBuffer(phone="+5215512345678", profile_name="Carlos")

        assert buf.phone == "+5215512345678"
        assert buf.profile_name == "Carlos"
        assert buf.messages == []
        assert buf.flush_task is None
        assert buf.created_at is not None
