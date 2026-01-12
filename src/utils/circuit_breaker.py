"""
Circuit Breaker for LLM Graceful Degradation

Implements the circuit breaker pattern to handle OpenAI outages gracefully.
When failures exceed threshold, circuit opens and returns fallback responses
immediately without hitting the API.
"""

import asyncio
from enum import Enum
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional, Callable, TypeVar, Awaitable, Any
from src.config import get_settings
from src.utils.observability import logger

T = TypeVar("T")


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation, requests flow through
    OPEN = "open"          # Failing, reject requests immediately
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitStats:
    """Circuit breaker statistics."""
    consecutive_failures: int = 0
    total_failures: int = 0
    total_successes: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    opened_at: Optional[datetime] = None
    state_changes: int = 0


class CircuitOpenError(Exception):
    """Raised when circuit is open and request is rejected."""
    pass


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures.

    States:
    - CLOSED: Normal operation. Failures are counted.
    - OPEN: Service is down. Requests fail fast with fallback.
    - HALF_OPEN: Testing recovery. Limited requests allowed.

    Usage:
        breaker = CircuitBreaker(name="openai")

        try:
            result = await breaker.call(
                func=call_openai,
                fallback=lambda: default_response
            )
        except CircuitOpenError:
            # Circuit is open, fallback was returned
            pass
    """

    def __init__(
        self,
        name: str,
        failure_threshold: Optional[int] = None,
        recovery_timeout: Optional[float] = None,
        half_open_max_calls: Optional[int] = None,
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Identifier for this circuit (for logging)
            failure_threshold: Consecutive failures before opening
            recovery_timeout: Seconds before trying recovery
            half_open_max_calls: Max calls allowed in half-open state
        """
        settings = get_settings()
        self.name = name
        self._failure_threshold = failure_threshold or settings.circuit_breaker_failure_threshold
        self._recovery_timeout = recovery_timeout or settings.circuit_breaker_recovery_timeout
        self._half_open_max_calls = half_open_max_calls or settings.circuit_breaker_half_open_max_calls

        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self._state

    @property
    def stats(self) -> CircuitStats:
        """Circuit statistics."""
        return self._stats

    async def call(
        self,
        func: Callable[[], Awaitable[T]],
        fallback: Callable[[], T],
    ) -> T:
        """
        Execute function through circuit breaker.

        Args:
            func: Async function to execute
            fallback: Function returning fallback value if circuit open

        Returns:
            Result from func or fallback

        Raises:
            CircuitOpenError: When circuit is open (after returning fallback)
        """
        async with self._lock:
            await self._check_state_transition()

            if self._state == CircuitState.OPEN:
                logger.warning(f"Circuit '{self.name}' is OPEN, using fallback")
                return fallback()

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self._half_open_max_calls:
                    logger.warning(f"Circuit '{self.name}' HALF_OPEN limit reached, using fallback")
                    return fallback()
                self._half_open_calls += 1

        # Execute outside lock to allow concurrency
        try:
            result = await func()
            await self._record_success()
            return result
        except Exception as e:
            await self._record_failure(e)
            raise

    async def call_with_fallback(
        self,
        func: Callable[[], Awaitable[T]],
        fallback: Callable[[], T],
    ) -> T:
        """
        Execute function, returning fallback on any failure.

        Unlike call(), this catches exceptions and returns fallback.

        Args:
            func: Async function to execute
            fallback: Function returning fallback value

        Returns:
            Result from func or fallback (never raises)
        """
        try:
            return await self.call(func, fallback)
        except Exception as e:
            logger.error(f"Circuit '{self.name}' call failed, using fallback: {e}")
            return fallback()

    async def _check_state_transition(self) -> None:
        """Check if state should transition based on time."""
        if self._state != CircuitState.OPEN:
            return

        if not self._stats.opened_at:
            return

        elapsed = (datetime.now(timezone.utc) - self._stats.opened_at).total_seconds()
        if elapsed >= self._recovery_timeout:
            self._transition_to(CircuitState.HALF_OPEN)
            self._half_open_calls = 0

    async def _record_success(self) -> None:
        """Record successful call."""
        async with self._lock:
            self._stats.consecutive_failures = 0
            self._stats.total_successes += 1
            self._stats.last_success_time = datetime.now(timezone.utc)

            if self._state == CircuitState.HALF_OPEN:
                logger.info(f"Circuit '{self.name}' recovered, closing")
                self._transition_to(CircuitState.CLOSED)

    async def _record_failure(self, error: Exception) -> None:
        """Record failed call."""
        async with self._lock:
            self._stats.consecutive_failures += 1
            self._stats.total_failures += 1
            self._stats.last_failure_time = datetime.now(timezone.utc)

            logger.warning(
                f"Circuit '{self.name}' failure {self._stats.consecutive_failures}/{self._failure_threshold}: {error}"
            )

            if self._state == CircuitState.HALF_OPEN:
                logger.warning(f"Circuit '{self.name}' probe failed, reopening")
                self._transition_to(CircuitState.OPEN)
            elif self._stats.consecutive_failures >= self._failure_threshold:
                logger.error(f"Circuit '{self.name}' threshold reached, opening")
                self._transition_to(CircuitState.OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to new state."""
        old_state = self._state
        self._state = new_state
        self._stats.state_changes += 1

        if new_state == CircuitState.OPEN:
            self._stats.opened_at = datetime.now(timezone.utc)

        logger.info(f"Circuit '{self.name}' state: {old_state.value} -> {new_state.value}")

    async def reset(self) -> None:
        """Manually reset circuit to closed state."""
        async with self._lock:
            self._transition_to(CircuitState.CLOSED)
            self._stats.consecutive_failures = 0
            self._half_open_calls = 0

    async def force_open(self) -> None:
        """Manually open circuit (for testing/maintenance)."""
        async with self._lock:
            self._transition_to(CircuitState.OPEN)

    def get_status(self) -> dict:
        """Get circuit status for monitoring."""
        return {
            "name": self.name,
            "state": self._state.value,
            "consecutive_failures": self._stats.consecutive_failures,
            "total_failures": self._stats.total_failures,
            "total_successes": self._stats.total_successes,
            "failure_threshold": self._failure_threshold,
            "recovery_timeout_seconds": self._recovery_timeout,
            "last_failure": self._stats.last_failure_time.isoformat() if self._stats.last_failure_time else None,
            "last_success": self._stats.last_success_time.isoformat() if self._stats.last_success_time else None,
            "opened_at": self._stats.opened_at.isoformat() if self._stats.opened_at else None,
        }


# Global circuit breaker instance for OpenAI
_openai_circuit: Optional[CircuitBreaker] = None


def get_openai_circuit() -> CircuitBreaker:
    """Get or create the OpenAI circuit breaker singleton."""
    global _openai_circuit
    if _openai_circuit is None:
        _openai_circuit = CircuitBreaker(name="openai")
    return _openai_circuit
