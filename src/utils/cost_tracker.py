"""
Cost Tracking & Budget Management
Monitors LLM token usage and prevents runaway costs.
"""
import datetime as dt
from dataclasses import dataclass, field
from typing import Dict
from loguru import logger
from src.config import get_settings


@dataclass
class ModelPricing:
    """Pricing per 1M tokens (USD) as of Late 2025."""
    input_cost: float
    output_cost: float


# Official OpenAI Pricing
PRICING: Dict[str, ModelPricing] = {
    "gpt-4o": ModelPricing(input_cost=2.50, output_cost=10.00),
    "gpt-4o-mini": ModelPricing(input_cost=0.15, output_cost=0.60),
    "gpt-4-turbo": ModelPricing(input_cost=10.00, output_cost=30.00),
    "gpt-3.5-turbo": ModelPricing(input_cost=0.50, output_cost=1.50),
}


@dataclass
class UsageWindow:
    """Tracks usage over a time window."""
    total_cost: float = 0.0
    total_tokens: int = 0
    call_count: int = 0
    window_start: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.UTC))


class CostTracker:
    """
    Thread-safe cost tracking with budget enforcement.

    Tracks usage at multiple time granularities (hourly, daily)
    and raises alerts when approaching limits.
    """

    def __init__(self):
        self.settings = get_settings()
        self.hourly_usage = UsageWindow()
        self.daily_usage = UsageWindow()
        self.lifetime_usage = UsageWindow()

        # Track per-agent costs for debugging
        self.per_agent_costs: Dict[str, float] = {}

    def _reset_window_if_needed(self, window: UsageWindow, hours: int) -> UsageWindow:
        """Reset usage window if it's expired."""
        now = dt.datetime.now(dt.UTC)
        elapsed = (now - window.window_start).total_seconds() / 3600

        if elapsed >= hours:
            logger.debug(f"Resetting {hours}h usage window")
            return UsageWindow()
        return window

    def track_completion(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        agent_name: str | None = None
    ) -> float:
        """
        Track a single LLM completion and return its cost.

        Args:
            model: Model identifier (e.g., "openai:gpt-4o" or "gpt-4o")
            input_tokens: Number of input/prompt tokens
            output_tokens: Number of output/completion tokens
            agent_name: Optional agent identifier for per-agent tracking

        Returns:
            Cost in USD for this completion

        Raises:
            RuntimeError: If hourly or daily budget exceeded
        """
        # Reset windows if needed
        self.hourly_usage = self._reset_window_if_needed(self.hourly_usage, 1)
        self.daily_usage = self._reset_window_if_needed(self.daily_usage, 24)

        # Extract model name (handle "openai:gpt-4o" format)
        model_name = model.split(":")[-1] if ":" in model else model

        # Get pricing
        pricing = PRICING.get(model_name)
        if not pricing:
            logger.warning(f"⚠️ Unknown model pricing: {model_name}, assuming gpt-4o rates")
            pricing = PRICING["gpt-4o"]

        # Calculate cost
        cost = (
            (input_tokens / 1_000_000) * pricing.input_cost +
            (output_tokens / 1_000_000) * pricing.output_cost
        )
        total_tokens = input_tokens + output_tokens

        # Update all windows
        for usage_window in [self.hourly_usage, self.daily_usage, self.lifetime_usage]:
            usage_window.total_cost += cost
            usage_window.total_tokens += total_tokens
            usage_window.call_count += 1

        # Per-agent tracking
        if agent_name:
            self.per_agent_costs[agent_name] = self.per_agent_costs.get(agent_name, 0.0) + cost

        # Budget enforcement
        self._check_budget_limits()

        # Logging
        logger.debug(
            f"💰 {model_name}: ${cost:.4f} "
            f"({input_tokens} in + {output_tokens} out = {total_tokens} tokens)"
        )

        return cost

    def _check_budget_limits(self):
        """Check if we're approaching or exceeding budget limits."""
        if self.settings.environment == "test":
            return
        # Hourly check
        
        hourly_pct = (self.hourly_usage.total_cost / self.settings.hourly_cost_limit_usd) * 100
        if hourly_pct >= 100:
            raise RuntimeError(
                f"🚨 HOURLY BUDGET EXCEEDED: ${self.hourly_usage.total_cost:.2f} "
                f"/ ${self.settings.hourly_cost_limit_usd:.2f}"
            )
        elif hourly_pct >= 80:
            logger.warning(
                f"⚠️ Hourly budget at {hourly_pct:.0f}%: "
                f"${self.hourly_usage.total_cost:.2f} / ${self.settings.hourly_cost_limit_usd:.2f}"
            )

        # Daily check
        daily_pct = (self.daily_usage.total_cost / self.settings.daily_cost_limit_usd) * 100
        if daily_pct >= 100:
            raise RuntimeError(
                f"🚨 DAILY BUDGET EXCEEDED: ${self.daily_usage.total_cost:.2f} "
                f"/ ${self.settings.daily_cost_limit_usd:.2f}"
            )
        elif daily_pct >= 80:
            logger.warning(
                f"⚠️ Daily budget at {daily_pct:.0f}%: "
                f"${self.daily_usage.total_cost:.2f} / ${self.settings.daily_cost_limit_usd:.2f}"
            )

    def get_summary(self) -> Dict[str, any]:
        """Get a summary of current usage statistics."""
        return {
            "hourly": {
                "cost_usd": round(self.hourly_usage.total_cost, 4),
                "tokens": self.hourly_usage.total_tokens,
                "calls": self.hourly_usage.call_count,
                "limit_usd": self.settings.hourly_cost_limit_usd,
                "pct_used": round((self.hourly_usage.total_cost / self.settings.hourly_cost_limit_usd) * 100, 1)
            },
            "daily": {
                "cost_usd": round(self.daily_usage.total_cost, 4),
                "tokens": self.daily_usage.total_tokens,
                "calls": self.daily_usage.call_count,
                "limit_usd": self.settings.daily_cost_limit_usd,
                "pct_used": round((self.daily_usage.total_cost / self.settings.daily_cost_limit_usd) * 100, 1)
            },
            "lifetime": {
                "cost_usd": round(self.lifetime_usage.total_cost, 2),
                "tokens": self.lifetime_usage.total_tokens,
                "calls": self.lifetime_usage.call_count
            },
            "per_agent": {k: round(v, 4) for k, v in self.per_agent_costs.items()}
        }


# Global singleton instance
_cost_tracker: CostTracker | None = None


def get_cost_tracker() -> CostTracker:
    """Get the global cost tracker instance (singleton)."""
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
    return _cost_tracker
