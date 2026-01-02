"""
Tests for LLM client retry logic and error handling.
Verifies exponential backoff, error categorization, and fallback behavior.
"""
import pytest
from unittest.mock import AsyncMock, Mock, patch
from src.utils.llm_client import (
    run_agent_with_retry,
    run_agent_with_fallback,
    LLMError,
    LLMCriticalError
)


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
