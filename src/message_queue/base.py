"""
Base Queue Interface

Abstract interface for message queues with retry logic and metrics.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class MessageStatus(str, Enum):
    """Message processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"


class QueuedMessage(BaseModel):
    """
    Message in the queue.

    Attributes:
        id: Unique message identifier
        phone: Lead phone number (E.164 format)
        body: Message content
        profile_name: WhatsApp profile name
        message_sid: Twilio message SID
        status: Current processing status
        retry_count: Number of retry attempts
        max_retries: Maximum retry attempts before dead letter
        created_at: Timestamp when message was queued
        scheduled_at: When to process (for retry delays)
        error: Last error message if failed
    """
    model_config = ConfigDict(use_enum_values=True)

    id: str
    phone: str
    body: str
    profile_name: Optional[str] = None
    message_sid: str
    status: MessageStatus = MessageStatus.PENDING
    retry_count: int = 0
    max_retries: int = 5
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    scheduled_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error: Optional[str] = None


class QueueMetrics(BaseModel):
    """
    Queue performance metrics.

    Attributes:
        pending: Number of messages awaiting processing
        processing: Number of messages currently being processed
        completed: Total successful messages
        failed: Total failed messages
        dead_letter: Messages in dead letter queue
        avg_processing_time_ms: Average processing duration
        error_rate: Percentage of failed messages
    """
    pending: int = 0
    processing: int = 0
    completed: int = 0
    failed: int = 0
    dead_letter: int = 0
    avg_processing_time_ms: float = 0.0
    error_rate: float = 0.0


class MessageQueue(ABC):
    """
    Abstract message queue interface.

    Implementations must provide:
    - Enqueue: Add message to queue
    - Dequeue: Get next message to process
    - Complete: Mark message as successfully processed
    - Fail: Handle failed message with retry logic
    - Metrics: Get current queue statistics
    """

    @abstractmethod
    async def enqueue(self, message: QueuedMessage) -> str:
        """
        Add message to queue.

        Args:
            message: Message to enqueue

        Returns:
            Message ID
        """
        pass

    @abstractmethod
    async def dequeue(self) -> Optional[QueuedMessage]:
        """
        Get next message to process.

        Returns:
            Next message or None if queue is empty
        """
        pass

    @abstractmethod
    async def complete(self, message_id: str) -> None:
        """
        Mark message as successfully processed.

        Args:
            message_id: ID of completed message
        """
        pass

    @abstractmethod
    async def fail(self, message_id: str, error: str) -> None:
        """
        Handle failed message with retry logic.

        Implements exponential backoff:
        - Retry 1: 1 minute
        - Retry 2: 5 minutes
        - Retry 3: 15 minutes
        - Retry 4: 1 hour
        - Retry 5: 6 hours
        - After max retries: Move to dead letter queue

        Args:
            message_id: ID of failed message
            error: Error description
        """
        pass

    @abstractmethod
    async def get_metrics(self) -> QueueMetrics:
        """
        Get current queue metrics.

        Returns:
            Queue statistics
        """
        pass

    @abstractmethod
    async def get_dead_letter_messages(self, limit: int = 100) -> list[QueuedMessage]:
        """
        Get messages in dead letter queue.

        Args:
            limit: Maximum messages to return

        Returns:
            List of dead letter messages
        """
        pass

    @abstractmethod
    async def retry_dead_letter(self, message_id: str) -> None:
        """
        Retry a message from dead letter queue.

        Resets retry count and moves back to pending.

        Args:
            message_id: ID of message to retry
        """
        pass
