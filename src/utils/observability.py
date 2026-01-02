"""
Structured Logging & Observability
Production-grade logging that's both human-readable and machine-parseable.
"""
import sys
from loguru import logger
from typing import Any, Dict
from src.config import get_settings


def configure_logging():
    """
    Configure loguru for production observability.

    In development: Human-readable colorized output
    In production: Structured JSON logs for ingestion (ELK, Datadog, etc.)
    """
    settings = get_settings()

    # Remove default handler
    logger.remove()

    # Development mode: Beautiful console output
    if not settings.enable_structured_logging:
        logger.add(
            sys.stderr,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "{message}"
            ),
            level=settings.log_level,
            colorize=True,
        )
    # Production mode: JSON structured logs
    else:
        logger.add(
            sys.stderr,
            format="{message}",
            level=settings.log_level,
            serialize=True,  # Output as JSON
        )

    logger.info(f"Logging configured: level={settings.log_level}, structured={settings.enable_structured_logging}")


def log_agent_execution(
    agent_name: str,
    lead_id: str,
    action: str,
    duration_ms: float | None = None,
    **context
):
    """
    Structured logging for agent executions.

    Args:
        agent_name: Name of the agent (e.g., "ClassifierAgent")
        lead_id: The lead being processed
        action: What action was performed (e.g., "classify", "decide_strategy")
        duration_ms: Execution time in milliseconds
        **context: Additional context (stage, intent, cost, etc.)

    Example:
        >>> log_agent_execution(
        ...     agent_name="DirectorAgent",
        ...     lead_id="+521...",
        ...     action="decide_strategy",
        ...     duration_ms=234.5,
        ...     stage="discovery",
        ...     intent="pricing"
        ... )
    """
    log_data = {
        "agent": agent_name,
        "lead_id": lead_id,
        "action": action,
    }

    if duration_ms is not None:
        log_data["duration_ms"] = round(duration_ms, 2)

    # Merge additional context
    log_data.update(context)

    # Bind the structured data and log
    logger.bind(**log_data).info(f"{agent_name} | {action}")


def log_llm_call(
    agent_name: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    duration_ms: float,
    success: bool = True,
    error: str | None = None
):
    """
    Structured logging for LLM API calls.

    Enables cost analysis, performance monitoring, and error tracking.

    Args:
        agent_name: Which agent made the call
        model: Model used (e.g., "gpt-4o")
        input_tokens: Input token count
        output_tokens: Output token count
        cost_usd: Cost of this call in USD
        duration_ms: API latency in milliseconds
        success: Whether the call succeeded
        error: Error message if failed
    """
    log_data = {
        "event_type": "llm_call",
        "agent": agent_name,
        "model": model,
        "tokens": {
            "input": input_tokens,
            "output": output_tokens,
            "total": input_tokens + output_tokens
        },
        "cost_usd": round(cost_usd, 6),
        "duration_ms": round(duration_ms, 2),
        "success": success
    }

    if error:
        log_data["error"] = error

    level = "info" if success else "error"
    logger.bind(**log_data).log(
        level.upper(),
        f"LLM Call: {model} | {input_tokens + output_tokens} tokens | ${cost_usd:.4f}"
    )


def log_business_event(
    event_type: str,
    lead_id: str,
    **details: Dict[str, Any]
):
    """
    Log business-critical events for analytics.

    Examples:
        - Lead stage transitions
        - Demo scheduled
        - Lead marked as lost

    Args:
        event_type: Type of event (e.g., "stage_transition", "demo_scheduled")
        lead_id: The lead involved
        **details: Event-specific data
    """
    log_data = {
        "event_type": event_type,
        "lead_id": lead_id,
        **details
    }

    logger.bind(**log_data).success(f"Business Event: {event_type}")
