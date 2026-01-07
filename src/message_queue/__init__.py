"""
Message Queue System

Provides async message queue processing for webhook handling with:
- Abstract queue interface supporting multiple backends
- In-memory queue for testing/MVP
- Redis queue for production
- Retry logic with exponential backoff
- Dead letter queue for failed messages
- Queue metrics and monitoring
"""

from src.message_queue.base import MessageQueue, QueuedMessage, QueueMetrics, MessageStatus
from src.message_queue.memory import InMemoryQueue
from src.message_queue.worker import QueueWorker

__all__ = [
    "MessageQueue",
    "QueuedMessage",
    "QueueMetrics",
    "MessageStatus",
    "InMemoryQueue",
    "QueueWorker",
]
