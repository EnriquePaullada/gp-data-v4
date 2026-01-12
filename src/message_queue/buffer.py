"""
Message Buffer for WhatsApp Burst Messages

Buffers incoming messages per lead for a configurable duration,
concatenating burst messages into a single pipeline trigger.
Handles the common WhatsApp pattern of users sending multiple
short messages that together form a single thought.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Awaitable, Optional
from src.config import get_settings
from src.utils.observability import logger


@dataclass
class BufferedMessage:
    """Individual message in the buffer."""
    body: str
    message_sid: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class LeadBuffer:
    """Buffer state for a single lead."""
    phone: str
    profile_name: Optional[str]
    messages: list[BufferedMessage] = field(default_factory=list)
    flush_task: Optional[asyncio.Task] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MessageBuffer:
    """
    Buffers WhatsApp messages per lead before processing.

    When a message arrives, it's added to a per-lead buffer.
    A timer starts (or resets) for that lead. When the timer
    expires without new messages, all buffered messages are
    concatenated and sent to the callback for processing.

    This handles the common pattern where users send:
        "Hi"
        "I saw your ad"
        "How much does it cost?"

    Instead of 3 pipeline runs, we get 1 run with the full context.

    Usage:
        buffer = MessageBuffer(on_flush=process_message)
        await buffer.add("phone", "body", "sid", "name")
        # ... after buffer_seconds with no new messages ...
        # on_flush is called with concatenated message
    """

    def __init__(
        self,
        on_flush: Callable[[str, str, str, Optional[str]], Awaitable[None]],
        buffer_seconds: Optional[float] = None,
        max_messages: Optional[int] = None,
        separator: Optional[str] = None,
    ):
        """
        Initialize message buffer.

        Args:
            on_flush: Async callback when buffer flushes.
                      Args: (phone, combined_body, first_message_sid, profile_name)
            buffer_seconds: Seconds to wait before flushing (default from settings)
            max_messages: Max messages before force flush (default from settings)
            separator: String to join messages (default from settings)
        """
        settings = get_settings()
        self._on_flush = on_flush
        self._buffer_seconds = buffer_seconds or settings.message_buffer_seconds
        self._max_messages = max_messages or settings.message_buffer_max_messages
        self._separator = separator if separator is not None else settings.message_buffer_separator
        self._buffers: dict[str, LeadBuffer] = {}
        self._lock = asyncio.Lock()

    async def add(
        self,
        phone: str,
        body: str,
        message_sid: str,
        profile_name: Optional[str] = None,
    ) -> None:
        """
        Add a message to the buffer.

        Resets the flush timer for this lead. If max_messages is
        reached, forces an immediate flush.

        Args:
            phone: Lead phone number (E.164 format)
            body: Message content
            message_sid: Twilio message SID
            profile_name: WhatsApp profile name
        """
        async with self._lock:
            if phone not in self._buffers:
                self._buffers[phone] = LeadBuffer(
                    phone=phone,
                    profile_name=profile_name,
                )
                logger.debug(f"Created buffer for {phone}")

            lead_buffer = self._buffers[phone]

            # Cancel existing flush timer
            if lead_buffer.flush_task and not lead_buffer.flush_task.done():
                lead_buffer.flush_task.cancel()
                try:
                    await lead_buffer.flush_task
                except asyncio.CancelledError:
                    pass

            # Add message to buffer
            lead_buffer.messages.append(BufferedMessage(
                body=body,
                message_sid=message_sid,
            ))

            # Update profile name if provided
            if profile_name:
                lead_buffer.profile_name = profile_name

            message_count = len(lead_buffer.messages)
            logger.debug(
                f"Buffered message for {phone}",
                extra={"count": message_count, "max": self._max_messages}
            )

            # Force flush if max messages reached
            if message_count >= self._max_messages:
                logger.info(f"Max buffer size reached for {phone}, force flushing")
                await self._flush_lead(phone)
                return

            # Schedule new flush timer
            lead_buffer.flush_task = asyncio.create_task(
                self._scheduled_flush(phone)
            )

    async def _scheduled_flush(self, phone: str) -> None:
        """
        Wait for buffer duration then flush.

        Args:
            phone: Lead phone number to flush
        """
        try:
            await asyncio.sleep(self._buffer_seconds)

            async with self._lock:
                if phone in self._buffers:
                    await self._flush_lead(phone)
        except asyncio.CancelledError:
            # Timer was cancelled (new message arrived)
            pass
        except Exception as e:
            # Log but don't re-raise from background task
            logger.error(f"Scheduled flush failed for {phone}: {e}")

    async def _flush_lead(self, phone: str) -> None:
        """
        Flush buffered messages for a lead.

        Must be called while holding the lock.

        Args:
            phone: Lead phone number to flush
        """
        lead_buffer = self._buffers.pop(phone, None)
        if not lead_buffer or not lead_buffer.messages:
            return

        # Concatenate messages
        combined_body = self._separator.join(
            msg.body for msg in lead_buffer.messages
        )

        # Use first message's SID as the reference
        first_sid = lead_buffer.messages[0].message_sid

        message_count = len(lead_buffer.messages)
        logger.info(
            f"Flushing buffer for {phone}",
            extra={"message_count": message_count, "combined_length": len(combined_body)}
        )

        # Release lock before callback to prevent deadlocks
        # Copy values we need
        profile_name = lead_buffer.profile_name

        # Call flush callback outside lock
        try:
            await self._on_flush(phone, combined_body, first_sid, profile_name)
        except Exception as e:
            logger.error(f"Buffer flush callback failed for {phone}: {e}")
            raise

    async def flush_all(self) -> None:
        """
        Flush all pending buffers immediately.

        Useful for graceful shutdown.
        """
        async with self._lock:
            phones = list(self._buffers.keys())

        for phone in phones:
            async with self._lock:
                if phone in self._buffers:
                    # Cancel timer
                    lead_buffer = self._buffers[phone]
                    if lead_buffer.flush_task and not lead_buffer.flush_task.done():
                        lead_buffer.flush_task.cancel()
                        try:
                            await lead_buffer.flush_task
                        except asyncio.CancelledError:
                            pass
                    await self._flush_lead(phone)

    async def get_pending_count(self) -> int:
        """
        Get number of leads with pending buffered messages.

        Returns:
            Count of leads with pending messages
        """
        async with self._lock:
            return len(self._buffers)

    async def get_buffer_stats(self) -> dict:
        """
        Get buffer statistics.

        Returns:
            Dict with pending_leads and total_messages
        """
        async with self._lock:
            total_messages = sum(
                len(buf.messages) for buf in self._buffers.values()
            )
            return {
                "pending_leads": len(self._buffers),
                "total_buffered_messages": total_messages,
                "buffer_seconds": self._buffer_seconds,
                "max_messages_per_lead": self._max_messages,
            }
