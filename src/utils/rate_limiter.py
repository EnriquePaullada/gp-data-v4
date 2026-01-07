"""
Rate Limiting and Abuse Detection

Provides per-lead rate limiting with Redis backend and in-memory fallback.
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List
from dataclasses import dataclass
from collections import defaultdict
from loguru import logger


@dataclass
class RateLimitResult:
    """
    Result of rate limit check.

    Attributes:
        allowed: Whether the request is allowed
        remaining: Number of requests remaining in current window
        reset_at: When the rate limit window resets
        retry_after: Seconds to wait before retrying (if blocked)
        reason: Why the request was blocked (if applicable)
    """
    allowed: bool
    remaining: int
    reset_at: datetime
    retry_after: Optional[int] = None
    reason: Optional[str] = None


class RateLimiter(ABC):
    """Abstract rate limiter interface."""

    @abstractmethod
    async def check_rate_limit(self, lead_id: str) -> RateLimitResult:
        """
        Check if lead has exceeded rate limit.

        Args:
            lead_id: Phone number or lead identifier

        Returns:
            Rate limit result
        """
        pass

    @abstractmethod
    async def is_banned(self, lead_id: str) -> bool:
        """
        Check if lead is temporarily banned.

        Args:
            lead_id: Lead identifier

        Returns:
            True if banned, False otherwise
        """
        pass

    @abstractmethod
    async def ban_lead(self, lead_id: str, duration_seconds: int, reason: str) -> None:
        """
        Temporarily ban a lead.

        Args:
            lead_id: Lead identifier
            duration_seconds: Ban duration
            reason: Reason for ban
        """
        pass

    @abstractmethod
    async def detect_spike(self, lead_id: str) -> bool:
        """
        Detect sudden spike in message frequency.

        Args:
            lead_id: Lead identifier

        Returns:
            True if spike detected, False otherwise
        """
        pass


class InMemoryRateLimiter(RateLimiter):
    """
    In-memory rate limiter implementation.

    Uses sliding window algorithm for rate limiting.
    Suitable for single-instance deployments or testing.
    """

    def __init__(
        self,
        max_requests: int = 10,
        window_seconds: int = 3600,  # 1 hour
        spike_threshold: int = 5,  # 5 messages in spike_window
        spike_window_seconds: int = 60,  # 1 minute
        ban_duration_seconds: int = 3600,  # 1 hour
    ):
        """
        Initialize in-memory rate limiter.

        Args:
            max_requests: Max requests per window
            window_seconds: Time window in seconds
            spike_threshold: Number of requests that trigger spike detection
            spike_window_seconds: Time window for spike detection
            ban_duration_seconds: Default ban duration
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.spike_threshold = spike_threshold
        self.spike_window_seconds = spike_window_seconds
        self.ban_duration_seconds = ban_duration_seconds

        # Storage
        self._requests: Dict[str, List[datetime]] = defaultdict(list)
        self._bans: Dict[str, tuple[datetime, str]] = {}  # lead_id -> (ban_until, reason)
        self._lock = asyncio.Lock()

    async def check_rate_limit(self, lead_id: str) -> RateLimitResult:
        """
        Check if lead has exceeded rate limit.

        Uses sliding window algorithm - counts requests in the last N seconds.

        Args:
            lead_id: Lead identifier

        Returns:
            Rate limit result
        """
        async with self._lock:
            now = datetime.now(timezone.utc)
            window_start = now - timedelta(seconds=self.window_seconds)

            # Clean up old requests
            self._requests[lead_id] = [
                ts for ts in self._requests[lead_id]
                if ts > window_start
            ]

            # Count requests in current window
            request_count = len(self._requests[lead_id])

            if request_count >= self.max_requests:
                # Rate limit exceeded
                oldest_request = min(self._requests[lead_id])
                reset_at = oldest_request + timedelta(seconds=self.window_seconds)
                retry_after = int((reset_at - now).total_seconds())

                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_at=reset_at,
                    retry_after=max(retry_after, 1),
                    reason=f"Rate limit exceeded: {request_count}/{self.max_requests} requests"
                )

            # Add current request
            self._requests[lead_id].append(now)

            # Calculate reset time
            reset_at = now + timedelta(seconds=self.window_seconds)

            return RateLimitResult(
                allowed=True,
                remaining=self.max_requests - (request_count + 1),
                reset_at=reset_at
            )

    async def is_banned(self, lead_id: str) -> bool:
        """
        Check if lead is temporarily banned.

        Args:
            lead_id: Lead identifier

        Returns:
            True if banned, False otherwise
        """
        async with self._lock:
            if lead_id not in self._bans:
                return False

            ban_until, _ = self._bans[lead_id]
            now = datetime.now(timezone.utc)

            if now > ban_until:
                # Ban expired
                del self._bans[lead_id]
                return False

            return True

    async def ban_lead(self, lead_id: str, duration_seconds: int, reason: str) -> None:
        """
        Temporarily ban a lead.

        Args:
            lead_id: Lead identifier
            duration_seconds: Ban duration
            reason: Reason for ban
        """
        async with self._lock:
            ban_until = datetime.now(timezone.utc) + timedelta(seconds=duration_seconds)
            self._bans[lead_id] = (ban_until, reason)

            logger.warning(
                f"Lead {lead_id} banned",
                extra={
                    "lead_id": lead_id,
                    "ban_until": ban_until.isoformat(),
                    "reason": reason
                }
            )

    async def detect_spike(self, lead_id: str) -> bool:
        """
        Detect sudden spike in message frequency.

        A spike is detected if the lead sends spike_threshold messages
        within spike_window_seconds.

        Args:
            lead_id: Lead identifier

        Returns:
            True if spike detected, False otherwise
        """
        async with self._lock:
            now = datetime.now(timezone.utc)
            spike_window_start = now - timedelta(seconds=self.spike_window_seconds)

            # Count requests in spike window
            recent_requests = [
                ts for ts in self._requests.get(lead_id, [])
                if ts > spike_window_start
            ]

            spike_detected = len(recent_requests) >= self.spike_threshold

            if spike_detected:
                logger.warning(
                    f"Spike detected for lead {lead_id}",
                    extra={
                        "lead_id": lead_id,
                        "request_count": len(recent_requests),
                        "threshold": self.spike_threshold,
                        "window_seconds": self.spike_window_seconds
                    }
                )

            return spike_detected

    async def get_ban_info(self, lead_id: str) -> Optional[tuple[datetime, str]]:
        """
        Get ban information for a lead.

        Args:
            lead_id: Lead identifier

        Returns:
            Tuple of (ban_until, reason) or None if not banned
        """
        async with self._lock:
            if lead_id in self._bans:
                ban_until, reason = self._bans[lead_id]
                if datetime.now(timezone.utc) <= ban_until:
                    return (ban_until, reason)
                else:
                    del self._bans[lead_id]
            return None
