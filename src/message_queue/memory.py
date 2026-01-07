"""
In-Memory Message Queue

Simple in-memory queue implementation for testing and MVP deployments.
Uses asyncio primitives for thread-safe async operations.
"""

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.message_queue.base import (
    MessageQueue,
    QueuedMessage,
    QueueMetrics,
    MessageStatus,
)


class InMemoryQueue(MessageQueue):
    """
    In-memory message queue implementation.

    Uses asyncio.Queue for thread-safe async operations.
    Stores messages in memory dictionaries - data is lost on restart.

    Suitable for:
    - Testing
    - MVP deployments
    - Single-instance applications

    Not suitable for:
    - Production multi-instance deployments
    - High availability requirements
    - Long-term message persistence
    """

    def __init__(self):
        """Initialize in-memory queue."""
        self._messages: dict[str, QueuedMessage] = {}
        self._pending_queue: asyncio.Queue = asyncio.Queue()
        self._processing: set[str] = set()
        self._completed: set[str] = set()
        self._failed: set[str] = set()
        self._dead_letter: dict[str, QueuedMessage] = {}
        self._processing_times: list[float] = []
        self._lock = asyncio.Lock()

        # Retry delay schedule (in seconds)
        self._retry_delays = [
            60,        # 1 minute
            300,       # 5 minutes
            900,       # 15 minutes
            3600,      # 1 hour
            21600,     # 6 hours
        ]

    async def enqueue(self, message: QueuedMessage) -> str:
        """
        Add message to queue.

        Args:
            message: Message to enqueue

        Returns:
            Message ID
        """
        async with self._lock:
            # Generate ID if not provided
            if not message.id:
                message.id = str(uuid.uuid4())

            # Store message
            self._messages[message.id] = message

            # Add to pending queue
            await self._pending_queue.put(message.id)

            return message.id

    async def dequeue(self) -> Optional[QueuedMessage]:
        """
        Get next message to process.

        Checks if message is ready based on scheduled_at timestamp.
        If message is not ready yet (retry delay), puts it back in queue.

        Returns:
            Next message or None if queue is empty
        """
        try:
            # Get next message ID (non-blocking)
            message_id = await asyncio.wait_for(
                self._pending_queue.get(),
                timeout=0.1
            )
        except asyncio.TimeoutError:
            return None

        async with self._lock:
            message = self._messages.get(message_id)
            if not message:
                return None

            # Check if message is ready to process
            now = datetime.now(timezone.utc)
            if message.scheduled_at > now:
                # Not ready yet, re-queue
                await self._pending_queue.put(message_id)
                return None

            # Mark as processing
            message.status = MessageStatus.PROCESSING
            self._processing.add(message_id)

            return message

    async def complete(self, message_id: str) -> None:
        """
        Mark message as successfully processed.

        Args:
            message_id: ID of completed message
        """
        async with self._lock:
            message = self._messages.get(message_id)
            if not message:
                return

            # Update status
            message.status = MessageStatus.COMPLETED

            # Move from processing to completed
            self._processing.discard(message_id)
            self._completed.add(message_id)

            # Track processing time
            processing_time = (
                datetime.now(timezone.utc) - message.created_at
            ).total_seconds() * 1000
            self._processing_times.append(processing_time)

            # Keep only last 1000 processing times
            if len(self._processing_times) > 1000:
                self._processing_times = self._processing_times[-1000:]

    async def fail(self, message_id: str, error: str) -> None:
        """
        Handle failed message with retry logic.

        Implements exponential backoff. After max retries, moves to dead letter queue.

        Args:
            message_id: ID of failed message
            error: Error description
        """
        async with self._lock:
            message = self._messages.get(message_id)
            if not message:
                return

            # Update error info
            message.error = error
            message.retry_count += 1

            # Remove from processing
            self._processing.discard(message_id)

            # Check if we should retry
            if message.retry_count <= message.max_retries:
                # Calculate retry delay
                delay_index = min(message.retry_count - 1, len(self._retry_delays) - 1)
                delay_seconds = self._retry_delays[delay_index]

                # Schedule retry
                message.scheduled_at = datetime.now(timezone.utc) + timedelta(
                    seconds=delay_seconds
                )
                message.status = MessageStatus.PENDING

                # Re-queue
                await self._pending_queue.put(message_id)
                self._failed.add(message_id)
            else:
                # Max retries exceeded, move to dead letter queue
                message.status = MessageStatus.DEAD_LETTER
                self._dead_letter[message_id] = message
                self._failed.add(message_id)

    async def get_metrics(self) -> QueueMetrics:
        """
        Get current queue metrics.

        Returns:
            Queue statistics
        """
        async with self._lock:
            total_messages = len(self._completed) + len(self._failed)
            error_rate = (
                (len(self._failed) / total_messages * 100)
                if total_messages > 0
                else 0.0
            )

            avg_time = (
                sum(self._processing_times) / len(self._processing_times)
                if self._processing_times
                else 0.0
            )

            return QueueMetrics(
                pending=self._pending_queue.qsize(),
                processing=len(self._processing),
                completed=len(self._completed),
                failed=len(self._failed),
                dead_letter=len(self._dead_letter),
                avg_processing_time_ms=avg_time,
                error_rate=error_rate,
            )

    async def get_dead_letter_messages(self, limit: int = 100) -> list[QueuedMessage]:
        """
        Get messages in dead letter queue.

        Args:
            limit: Maximum messages to return

        Returns:
            List of dead letter messages
        """
        async with self._lock:
            messages = list(self._dead_letter.values())
            return messages[:limit]

    async def retry_dead_letter(self, message_id: str) -> None:
        """
        Retry a message from dead letter queue.

        Resets retry count and moves back to pending.

        Args:
            message_id: ID of message to retry
        """
        async with self._lock:
            message = self._dead_letter.pop(message_id, None)
            if not message:
                return

            # Reset retry state
            message.retry_count = 0
            message.status = MessageStatus.PENDING
            message.scheduled_at = datetime.now(timezone.utc)
            message.error = None

            # Re-queue
            await self._pending_queue.put(message_id)
