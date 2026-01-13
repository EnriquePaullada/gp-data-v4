"""
LLM Client with Retry Logic & Error Handling
Provides resilient LLM execution with exponential backoff.
"""
import asyncio
from typing import TypeVar, Any
from loguru import logger
from pydantic_ai import Agent
from src.config import get_settings

# Type variable for generic agent output
T = TypeVar('T')


class LLMError(Exception):
    """Recoverable LLM errors that should trigger retries."""
    pass


class LLMCriticalError(Exception):
    """Non-recoverable errors (auth failure, invalid prompt, etc.)."""
    pass


async def run_agent_with_retry(
    agent: Agent,
    prompt: str,
    deps: Any = None,
    max_retries: int | None = None
) -> T:
    """
    Executes an agent with exponential backoff retry logic.

    Args:
        agent: The PydanticAI agent to run
        prompt: The prompt to send to the agent
        deps: Optional dependencies for the agent
        max_retries: Override default retry count from settings

    Returns:
        The agent's output (typed based on agent's output_type)

    Raises:
        LLMCriticalError: For non-recoverable failures
        LLMError: After max retries exhausted

    Example:
        >>> agent = Agent('openai:gpt-4o', output_type=MyResponse)
        >>> result = await run_agent_with_retry(agent, "Classify this")
        >>> print(result.intent)
    """
    settings = get_settings()
    max_attempts = max_retries or settings.max_retries
    min_wait = settings.retry_min_wait_seconds
    max_wait = settings.retry_max_wait_seconds

    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            logger.debug(f"LLM attempt {attempt}/{max_attempts}")

            # Execute the agent
            if deps is not None:
                result = await agent.run(prompt, deps=deps)
            else:
                result = await agent.run(prompt)

            return result.output

        except Exception as e:
            last_error = e
            error_msg = str(e).lower()

            # Categorize the error
            if "rate" in error_msg and "limit" in error_msg:
                logger.warning(f"⏱️ Rate limit hit (attempt {attempt}/{max_attempts})")
                error_type = "rate_limit"

            elif "timeout" in error_msg or "timed out" in error_msg:
                logger.warning(f"⏱️ Timeout (attempt {attempt}/{max_attempts})")
                error_type = "timeout"

            elif any(code in error_msg for code in ["500", "502", "503", "504"]):
                logger.warning(f"🔧 Server error (attempt {attempt}/{max_attempts})")
                error_type = "server_error"

            elif "authentication" in error_msg or "api key" in error_msg or "401" in error_msg:
                logger.error(f"🚨 Authentication failure: {e}")
                raise LLMCriticalError(f"Authentication failed: {e}") from e

            elif "invalid" in error_msg and "request" in error_msg:
                logger.error(f"🚨 Invalid request: {e}")
                raise LLMCriticalError(f"Invalid request: {e}") from e

            else:
                # Unknown error - treat as recoverable but log it
                logger.warning(f"⚠️ Unknown error (attempt {attempt}/{max_attempts}): {e}")
                error_type = "unknown"

            # If this was the last attempt, raise
            if attempt == max_attempts:
                logger.error(f"❌ Max retries ({max_attempts}) exhausted. Last error: {e}")
                raise LLMError(f"Failed after {max_attempts} attempts: {e}") from e

            # Calculate exponential backoff with jitter
            wait_time = min(min_wait * (2 ** (attempt - 1)), max_wait)
            # Add 20% jitter to prevent thundering herd
            import random
            wait_time = wait_time * (0.8 + 0.4 * random.random())

            logger.info(f"⏳ Retrying in {wait_time:.1f}s... (error: {error_type})")
            await asyncio.sleep(wait_time)

    # Should never reach here, but just in case
    raise LLMError(f"Unexpected retry loop exit. Last error: {last_error}")


async def run_agent_with_fallback(
    agent: Agent,
    prompt: str,
    fallback_factory: callable,
    deps: Any = None
) -> T:
    """
    Executes an agent with a fallback response if all retries fail.

    Useful for agents where a degraded response is better than total failure.

    Args:
        agent: The PydanticAI agent to run
        prompt: The prompt to send to the agent
        fallback_factory: Function that returns a safe default response
        deps: Optional dependencies for the agent

    Returns:
        Either the agent's output or the fallback response

    Example:
        >>> def safe_classification():
        >>>     return ClassifierResponse(intent=Intent.UNCLEAR, ...)
        >>> result = await run_agent_with_fallback(agent, prompt, safe_classification)
    """
    try:
        return await run_agent_with_retry(agent, prompt, deps=deps)
    except (LLMError, LLMCriticalError) as e:
        logger.error(f"🛟 LLM failed, using fallback response: {e}")
        return fallback_factory()
