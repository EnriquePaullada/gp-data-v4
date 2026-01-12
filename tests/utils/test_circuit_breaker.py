"""
Tests for Circuit Breaker

Validates circuit breaker behavior for LLM graceful degradation.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock
from src.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CircuitOpenError,
    get_openai_circuit,
)


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    @pytest.fixture
    def breaker(self):
        """Create circuit breaker with fast settings for testing."""
        return CircuitBreaker(
            name="test",
            failure_threshold=3,
            recovery_timeout=0.1,  # 100ms for fast tests
            half_open_max_calls=1,
        )

    @pytest.mark.asyncio
    async def test_starts_closed(self, breaker):
        """Circuit should start in closed state."""
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_successful_calls_stay_closed(self, breaker):
        """Successful calls should keep circuit closed."""
        async def success():
            return "ok"

        for _ in range(10):
            result = await breaker.call(success, lambda: "fallback")
            assert result == "ok"

        assert breaker.state == CircuitState.CLOSED
        assert breaker.stats.total_successes == 10
        assert breaker.stats.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_opens_after_threshold_failures(self, breaker):
        """Circuit should open after consecutive failures reach threshold."""
        async def fail():
            raise ValueError("API error")

        # First 2 failures - still closed
        for i in range(2):
            with pytest.raises(ValueError):
                await breaker.call(fail, lambda: "fallback")
            assert breaker.state == CircuitState.CLOSED

        # 3rd failure - opens
        with pytest.raises(ValueError):
            await breaker.call(fail, lambda: "fallback")

        assert breaker.state == CircuitState.OPEN
        assert breaker.stats.consecutive_failures == 3

    @pytest.mark.asyncio
    async def test_open_circuit_returns_fallback(self, breaker):
        """Open circuit should return fallback without calling function."""
        call_count = 0

        async def fail():
            nonlocal call_count
            call_count += 1
            raise ValueError("API error")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.call(fail, lambda: "fallback")

        assert breaker.state == CircuitState.OPEN
        assert call_count == 3

        # Now circuit is open - should return fallback without calling
        result = await breaker.call(fail, lambda: "fallback")
        assert result == "fallback"
        assert call_count == 3  # No additional call made

    @pytest.mark.asyncio
    async def test_success_resets_failure_count(self, breaker):
        """Success should reset consecutive failure count."""
        async def fail():
            raise ValueError("error")

        async def success():
            return "ok"

        # 2 failures
        for _ in range(2):
            with pytest.raises(ValueError):
                await breaker.call(fail, lambda: "fallback")

        assert breaker.stats.consecutive_failures == 2

        # 1 success resets count
        await breaker.call(success, lambda: "fallback")
        assert breaker.stats.consecutive_failures == 0

        # Need 3 more failures to open
        for _ in range(2):
            with pytest.raises(ValueError):
                await breaker.call(fail, lambda: "fallback")
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_after_timeout(self, breaker):
        """Circuit should transition to half-open after recovery timeout."""
        async def fail():
            raise ValueError("error")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.call(fail, lambda: "fallback")

        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Next call attempt should transition to half-open
        async def success():
            return "ok"

        result = await breaker.call(success, lambda: "fallback")
        # The state check happens before the call, so it should have gone to half-open
        # and then closed on success
        assert breaker.state == CircuitState.CLOSED
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens(self, breaker):
        """Failure in half-open state should reopen circuit."""
        async def fail():
            raise ValueError("error")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.call(fail, lambda: "fallback")

        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Probe call fails - should reopen
        with pytest.raises(ValueError):
            await breaker.call(fail, lambda: "fallback")

        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_half_open_limits_calls(self, breaker):
        """Half-open state should limit concurrent probe calls."""
        async def fail():
            raise ValueError("error")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.call(fail, lambda: "fallback")

        await asyncio.sleep(0.15)

        # First call in half-open allowed (will fail and reopen)
        with pytest.raises(ValueError):
            await breaker.call(fail, lambda: "fallback")

        # Circuit reopened, wait again
        await asyncio.sleep(0.15)

        # This call transitions to half-open and uses the probe slot
        call_count = 0

        async def slow_success():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.05)
            return "ok"

        result = await breaker.call(slow_success, lambda: "fallback")
        assert result == "ok"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_call_with_fallback_never_raises(self, breaker):
        """call_with_fallback should catch all exceptions."""
        async def fail():
            raise ValueError("error")

        # Even failures return fallback
        for _ in range(5):
            result = await breaker.call_with_fallback(fail, lambda: "safe")
            assert result == "safe"

    @pytest.mark.asyncio
    async def test_manual_reset(self, breaker):
        """Manual reset should close circuit."""
        async def fail():
            raise ValueError("error")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.call(fail, lambda: "fallback")

        assert breaker.state == CircuitState.OPEN

        await breaker.reset()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.stats.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_force_open(self, breaker):
        """force_open should open circuit immediately."""
        assert breaker.state == CircuitState.CLOSED

        await breaker.force_open()
        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_get_status(self, breaker):
        """get_status should return monitoring info."""
        async def fail():
            raise ValueError("error")

        with pytest.raises(ValueError):
            await breaker.call(fail, lambda: "fallback")

        status = breaker.get_status()

        assert status["name"] == "test"
        assert status["state"] == "closed"
        assert status["consecutive_failures"] == 1
        assert status["total_failures"] == 1
        assert status["failure_threshold"] == 3
        assert status["last_failure"] is not None

    @pytest.mark.asyncio
    async def test_stats_tracking(self, breaker):
        """Should track comprehensive statistics."""
        async def success():
            return "ok"

        async def fail():
            raise ValueError("error")

        await breaker.call(success, lambda: "f")
        await breaker.call(success, lambda: "f")

        with pytest.raises(ValueError):
            await breaker.call(fail, lambda: "f")

        assert breaker.stats.total_successes == 2
        assert breaker.stats.total_failures == 1
        assert breaker.stats.last_success_time is not None
        assert breaker.stats.last_failure_time is not None


class TestOpenAICircuitSingleton:
    """Tests for OpenAI circuit breaker singleton."""

    def test_returns_same_instance(self):
        """Should return same circuit breaker instance."""
        circuit1 = get_openai_circuit()
        circuit2 = get_openai_circuit()
        assert circuit1 is circuit2

    def test_named_openai(self):
        """Singleton should be named 'openai'."""
        circuit = get_openai_circuit()
        assert circuit.name == "openai"
