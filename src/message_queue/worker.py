"""
Queue Worker

Background worker that processes messages from the queue.
"""

import asyncio
from typing import Callable, Awaitable
from loguru import logger

from src.message_queue.base import MessageQueue, QueuedMessage


class QueueWorker:
    """
    Background worker for processing queued messages.

    Continuously polls the queue for new messages and processes them
    using the provided handler function.

    Attributes:
        queue: Message queue to process
        handler: Async function to process each message
        max_concurrent: Maximum number of concurrent message processors
        poll_interval: Seconds to wait between queue polls
    """

    def __init__(
        self,
        queue: MessageQueue,
        handler: Callable[[QueuedMessage], Awaitable[None]],
        max_concurrent: int = 10,
        poll_interval: float = 1.0,
    ):
        """
        Initialize queue worker.

        Args:
            queue: Message queue to process
            handler: Async function that processes messages
            max_concurrent: Max concurrent message processors
            poll_interval: Seconds between queue polls
        """
        self.queue = queue
        self.handler = handler
        self.max_concurrent = max_concurrent
        self.poll_interval = poll_interval
        self._running = False
        self._tasks: set[asyncio.Task] = set()
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def start(self) -> None:
        """
        Start the worker.

        Begins polling the queue and processing messages.
        Runs until stop() is called.
        """
        if self._running:
            logger.warning("Worker already running")
            return

        self._running = True
        logger.info(
            f"🚀 Queue worker started (max_concurrent={self.max_concurrent}, "
            f"poll_interval={self.poll_interval}s)"
        )

        try:
            while self._running:
                # Get next message from queue
                message = await self.queue.dequeue()

                if message:
                    # Process message in background task
                    task = asyncio.create_task(self._process_message(message))
                    self._tasks.add(task)
                    task.add_done_callback(self._tasks.discard)
                else:
                    # No messages available, wait before polling again
                    await asyncio.sleep(self.poll_interval)

                # Clean up completed tasks
                await self._cleanup_tasks()

        except Exception as e:
            logger.error(f"Worker crashed: {e}", exc_info=True)
            raise

        finally:
            logger.info("🛑 Queue worker stopped")

    async def stop(self) -> None:
        """
        Stop the worker.

        Gracefully shuts down:
        1. Stops accepting new messages
        2. Waits for in-flight messages to complete
        3. Cancels any remaining tasks
        """
        if not self._running:
            return

        logger.info("Stopping queue worker...")
        self._running = False

        # Wait for all tasks to complete (with timeout)
        if self._tasks:
            logger.info(f"Waiting for {len(self._tasks)} tasks to complete...")
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._tasks, return_exceptions=True),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for tasks, cancelling remaining")
                for task in self._tasks:
                    task.cancel()

    async def _process_message(self, message: QueuedMessage) -> None:
        """
        Process a single message with concurrency control.

        Args:
            message: Message to process
        """
        async with self._semaphore:
            try:
                logger.debug(
                    f"Processing message {message.id} (retry {message.retry_count})"
                )

                # Call the handler
                await self.handler(message)

                # Mark as completed
                await self.queue.complete(message.id)

                logger.info(
                    f"✅ Message {message.id} processed successfully",
                    extra={
                        "message_id": message.id,
                        "phone": message.phone,
                        "retry_count": message.retry_count,
                    }
                )

            except Exception as e:
                logger.error(
                    f"❌ Failed to process message {message.id}: {e}",
                    extra={
                        "message_id": message.id,
                        "phone": message.phone,
                        "retry_count": message.retry_count,
                        "error": str(e),
                    },
                    exc_info=True
                )

                # Mark as failed (will trigger retry logic)
                await self.queue.fail(message.id, str(e))

    async def _cleanup_tasks(self) -> None:
        """Remove completed tasks from the set."""
        done_tasks = {task for task in self._tasks if task.done()}
        self._tasks -= done_tasks
