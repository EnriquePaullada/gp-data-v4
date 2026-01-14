"""
Tests for LLM client retry logic and error handling.
Verifies exponential backoff, error categorization, and fallback behavior.
"""
import pytest
from unittest.mock import AsyncMock, Mock, patch
from src.utils.llm_client import (
    run_agent_with_retry,
    run_agent_with_fallback,
    run_agent_with_circuit_breaker,
    get_circuit_status,
    is_circuit_open,
    LLMError,
    LLMCriticalError
)
from src.utils.circuit_breaker import get_openai_circuit, CircuitState


class MockAgent:
    """Mock PydanticAI agent for testing."""

    def __init__(self, should_fail=False, failure_count=0, error_type="timeout"):
        self.should_fail = should_fail
        self.failure_count = failure_count
        self.error_type = error_type
        self._call_count = 0

    async def run(self, prompt, deps=None):
        """Mock run method that can simulate failures."""
        self._call_count += 1

        # Fail for the first N calls
        if self._call_count <= self.failure_count:
            if self.error_type == "rate_limit":
                raise Exception("Rate limit exceeded. Please try again later.")
            elif self.error_type == "timeout":
                raise Exception("Request timed out")
            elif self.error_type == "500":
                raise Exception("500 Internal Server Error")
            elif self.error_type == "auth":
                raise Exception("Authentication failed: Invalid API key")
            elif self.error_type == "invalid_request":
                raise Exception("Invalid request: Missing required field")

        # Success response
        mock_result = Mock()
        mock_result.output = "Success response"
        return mock_result


@pytest.mark.asyncio
class TestRetryLogic:
    """Test suite for retry logic."""

    async def test_success_on_first_try(self):
        """Verify no retries when first attempt succeeds."""
        agent = MockAgent(should_fail=False)

        result = await run_agent_with_retry(agent, "test prompt")

        assert result == "Success response"
        assert agent._call_count == 1

    async def test_success_after_transient_failure(self):
        """Verify retry succeeds after transient error."""
        # Fail once, then succeed
        agent = MockAgent(failure_count=1, error_type="timeout")

        result = await run_agent_with_retry(agent, "test prompt", max_retries=3)

        assert result == "Success response"
        assert agent._call_count == 2  # 1 failure + 1 success

    async def test_success_after_multiple_failures(self):
        """Verify retry succeeds after multiple transient errors."""
        # Fail twice, then succeed
        agent = MockAgent(failure_count=2, error_type="500")

        result = await run_agent_with_retry(agent, "test prompt", max_retries=3)

        assert result == "Success response"
        assert agent._call_count == 3  # 2 failures + 1 success

    async def test_exhausted_retries_raises_error(self):
        """Verify LLMError raised when all retries exhausted."""
        # Fail on all attempts
        agent = MockAgent(failure_count=10, error_type="timeout")

        with pytest.raises(LLMError, match="Failed after 3 attempts"):
            await run_agent_with_retry(agent, "test prompt", max_retries=3)

        assert agent._call_count == 3

    async def test_rate_limit_error_triggers_retry(self):
        """Verify rate limit errors trigger retries."""
        agent = MockAgent(failure_count=1, error_type="rate_limit")

        result = await run_agent_with_retry(agent, "test prompt", max_retries=3)

        assert result == "Success response"
        assert agent._call_count == 2

    async def test_authentication_error_critical(self):
        """Verify authentication errors are non-recoverable."""
        agent = MockAgent(failure_count=10, error_type="auth")

        with pytest.raises(LLMCriticalError, match="Authentication failed"):
            await run_agent_with_retry(agent, "test prompt", max_retries=3)

        # Should fail immediately without retries
        assert agent._call_count == 1

    async def test_invalid_request_error_critical(self):
        """Verify invalid request errors are non-recoverable."""
        agent = MockAgent(failure_count=10, error_type="invalid_request")

        with pytest.raises(LLMCriticalError, match="Invalid request"):
            await run_agent_with_retry(agent, "test prompt", max_retries=3)

        # Should fail immediately without retries
        assert agent._call_count == 1

    async def test_retry_with_deps(self):
        """Verify retry logic works with agent dependencies."""
        agent = MockAgent(failure_count=1, error_type="timeout")
        mock_deps = {"key": "value"}

        # Mock the agent's run method to accept deps
        original_run = agent.run
        async def run_with_deps(prompt, deps=None):
            assert deps == mock_deps
            return await original_run(prompt, deps)

        agent.run = run_with_deps

        result = await run_agent_with_retry(agent, "test prompt", deps=mock_deps, max_retries=3)

        assert result == "Success response"

    async def test_exponential_backoff_timing(self):
        """Verify exponential backoff increases wait time."""
        agent = MockAgent(failure_count=3, error_type="timeout")

        import time
        start_time = time.time()

        with pytest.raises(LLMError):
            # Should wait: ~2s + ~4s = ~6s total (with jitter)
            await run_agent_with_retry(agent, "test prompt", max_retries=3)

        elapsed_time = time.time() - start_time

        # Verify we actually waited (should be at least 4-5 seconds)
        assert elapsed_time >= 4.0, f"Expected wait time, got {elapsed_time}s"


@pytest.mark.asyncio
class TestFallbackBehavior:
    """Test suite for fallback response behavior."""

    async def test_returns_result_on_success(self):
        """Verify fallback not used when agent succeeds."""
        agent = MockAgent(should_fail=False)
        fallback_called = False

        def fallback_factory():
            nonlocal fallback_called
            fallback_called = True
            return "Fallback response"

        result = await run_agent_with_fallback(agent, "test prompt", fallback_factory)

        assert result == "Success response"
        assert not fallback_called

    async def test_returns_fallback_on_failure(self):
        """Verify fallback used when all retries fail."""
        agent = MockAgent(failure_count=10, error_type="timeout")

        def fallback_factory():
            return "Fallback response"

        result = await run_agent_with_fallback(agent, "test prompt", fallback_factory)

        assert result == "Fallback response"

    async def test_returns_fallback_on_critical_error(self):
        """Verify fallback used for critical errors."""
        agent = MockAgent(failure_count=10, error_type="auth")

        def fallback_factory():
            return "Fallback response"

        result = await run_agent_with_fallback(agent, "test prompt", fallback_factory)

        assert result == "Fallback response"

    async def test_fallback_with_complex_object(self):
        """Verify fallback can return complex objects."""
        agent = MockAgent(failure_count=10, error_type="timeout")

        class ComplexResponse:
            def __init__(self):
                self.status = "fallback"
                self.data = {"key": "value"}

        def fallback_factory():
            return ComplexResponse()

        result = await run_agent_with_fallback(agent, "test prompt", fallback_factory)

        assert isinstance(result, ComplexResponse)
        assert result.status == "fallback"
        assert result.data == {"key": "value"}


@pytest.mark.asyncio
class TestCircuitBreakerIntegration:
    """Test suite for circuit breaker integration with LLM calls."""

    async def test_circuit_breaker_success_on_first_try(self):
        """Verifies circuit breaker allows successful calls through."""
        # Reset circuit to clean state
        circuit = get_openai_circuit()
        await circuit.reset()

        agent = MockAgent(should_fail=False)

        def fallback_factory():
            return "Fallback response"

        result = await run_agent_with_circuit_breaker(agent, "test prompt", fallback_factory)

        assert result == "Success response"
        assert agent._call_count == 1
        assert circuit.state == CircuitState.CLOSED

    async def test_circuit_breaker_opens_after_threshold_failures(self):
        """Verifies circuit opens after consecutive failures exceed threshold."""
        # Reset circuit to clean state
        circuit = get_openai_circuit()
        await circuit.reset()

        # Create agent that always fails
        agent = MockAgent(failure_count=100, error_type="timeout")

        def fallback_factory():
            return "Fallback response"

        # Make 5 consecutive failing calls (default threshold)
        for _ in range(5):
            try:
                await run_agent_with_circuit_breaker(agent, "test prompt", fallback_factory)
            except:
                pass

        # Circuit should now be open
        assert circuit.state == CircuitState.OPEN

    async def test_circuit_breaker_returns_fallback_when_open(self):
        """Verifies circuit breaker returns fallback immediately when open."""
        # Reset and force circuit open
        circuit = get_openai_circuit()
        await circuit.reset()
        await circuit.force_open()

        agent = MockAgent(should_fail=False)
        fallback_called = False

        def fallback_factory():
            nonlocal fallback_called
            fallback_called = True
            return "Fallback response"

        result = await run_agent_with_circuit_breaker(agent, "test prompt", fallback_factory)

        # Should return fallback without calling agent
        assert result == "Fallback response"
        assert fallback_called is True
        assert agent._call_count == 0  # Agent never called
        assert circuit.state == CircuitState.OPEN

    async def test_circuit_breaker_uses_retry_logic_when_closed(self):
        """Verifies circuit breaker uses retry logic when closed."""
        # Reset circuit
        circuit = get_openai_circuit()
        await circuit.reset()

        # Agent fails once, then succeeds
        agent = MockAgent(failure_count=1, error_type="timeout")

        def fallback_factory():
            return "Fallback response"

        result = await run_agent_with_circuit_breaker(agent, "test prompt", fallback_factory)

        # Should succeed after retry
        assert result == "Success response"
        assert agent._call_count == 2  # 1 failure + 1 success
        assert circuit.state == CircuitState.CLOSED

    async def test_circuit_breaker_with_deps(self):
        """Verifies circuit breaker works with agent dependencies."""
        # Reset circuit
        circuit = get_openai_circuit()
        await circuit.reset()

        agent = MockAgent(should_fail=False)
        mock_deps = {"key": "value"}

        # Mock the agent's run method to accept deps
        original_run = agent.run
        async def run_with_deps(prompt, deps=None):
            assert deps == mock_deps
            return await original_run(prompt, deps)

        agent.run = run_with_deps

        def fallback_factory():
            return "Fallback response"

        result = await run_agent_with_circuit_breaker(
            agent, "test prompt", fallback_factory, deps=mock_deps
        )

        assert result == "Success response"

    async def test_circuit_breaker_shared_state(self):
        """Verifies circuit breaker shares state across multiple calls."""
        # Reset circuit
        circuit = get_openai_circuit()
        await circuit.reset()

        # Create two different agents
        agent1 = MockAgent(failure_count=100, error_type="timeout")
        agent2 = MockAgent(should_fail=False)

        def fallback_factory():
            return "Fallback response"

        # Fail with agent1 multiple times to open circuit
        for _ in range(5):
            try:
                await run_agent_with_circuit_breaker(agent1, "test prompt", fallback_factory)
            except:
                pass

        # Circuit should be open
        assert circuit.state == CircuitState.OPEN

        # Now try with agent2 - should still get fallback due to shared circuit
        result = await run_agent_with_circuit_breaker(agent2, "test prompt", fallback_factory)

        assert result == "Fallback response"
        assert agent2._call_count == 0  # Never called due to open circuit

    async def test_circuit_breaker_recovery_to_half_open(self):
        """Verifies circuit transitions to half-open for recovery testing."""
        # Reset circuit
        circuit = get_openai_circuit()
        await circuit.reset()

        # Force circuit open
        await circuit.force_open()
        assert circuit.state == CircuitState.OPEN

        # Manually transition to half-open (simulating timeout passage)
        circuit._state = CircuitState.HALF_OPEN
        circuit._half_open_calls = 0

        agent = MockAgent(should_fail=False)

        def fallback_factory():
            return "Fallback response"

        # Successful call in half-open should close circuit
        result = await run_agent_with_circuit_breaker(agent, "test prompt", fallback_factory)

        assert result == "Success response"
        assert circuit.state == CircuitState.CLOSED

    async def test_circuit_breaker_half_open_failure_reopens(self):
        """Verifies circuit reopens if half-open probe fails."""
        # Reset circuit
        circuit = get_openai_circuit()
        await circuit.reset()

        # Force circuit to half-open state
        await circuit.force_open()
        circuit._state = CircuitState.HALF_OPEN
        circuit._half_open_calls = 0

        # Agent that will fail
        agent = MockAgent(failure_count=100, error_type="timeout")

        def fallback_factory():
            return "Fallback response"

        # Failing call in half-open should reopen circuit
        result = await run_agent_with_circuit_breaker(agent, "test prompt", fallback_factory)

        assert result == "Fallback response"
        assert circuit.state == CircuitState.OPEN

    async def test_get_circuit_status(self):
        """Verifies circuit status can be retrieved for monitoring."""
        # Reset circuit
        circuit = get_openai_circuit()
        await circuit.reset()

        status = get_circuit_status()

        assert "state" in status
        assert "consecutive_failures" in status
        assert "total_failures" in status
        assert "total_successes" in status
        assert status["state"] == "closed"

    async def test_is_circuit_open_helper(self):
        """Verifies is_circuit_open helper function."""
        # Reset circuit
        circuit = get_openai_circuit()
        await circuit.reset()

        assert is_circuit_open() is False

        # Force open
        await circuit.force_open()
        assert is_circuit_open() is True

        # Reset
        await circuit.reset()
        assert is_circuit_open() is False

    async def test_circuit_breaker_language_aware_fallback(self):
        """Verifies circuit breaker works with language-aware fallbacks."""
        # Reset circuit
        circuit = get_openai_circuit()
        await circuit.reset()
        await circuit.force_open()

        agent = MockAgent(should_fail=False)

        # Spanish fallback
        def spanish_fallback():
            return "Respuesta de respaldo"

        result = await run_agent_with_circuit_breaker(agent, "test prompt", spanish_fallback)

        assert result == "Respuesta de respaldo"
        assert circuit.state == CircuitState.OPEN
