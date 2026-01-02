"""
Tests for cost tracking and budget enforcement.
Verifies token counting, cost calculations, and budget limits.
"""
import pytest
from src.utils.cost_tracker import CostTracker, PRICING, get_cost_tracker
from src.config import get_settings


class TestCostTracker:
    """Test suite for CostTracker functionality."""

    def test_cost_calculation_gpt4o(self):
        """Verify accurate cost calculation for gpt-4o."""
        tracker = CostTracker()

        # 1M input tokens + 1M output tokens
        cost = tracker.track_completion(
            model="gpt-4o",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            agent_name="TestAgent"
        )

        # Expected: $2.50 (input) + $10.00 (output) = $12.50
        assert cost == 12.50
        assert tracker.lifetime_usage.total_cost == 12.50
        assert tracker.lifetime_usage.total_tokens == 2_000_000

    def test_cost_calculation_gpt4o_mini(self):
        """Verify accurate cost calculation for gpt-4o-mini."""
        tracker = CostTracker()

        # 1M input tokens + 1M output tokens
        cost = tracker.track_completion(
            model="gpt-4o-mini",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            agent_name="TestAgent"
        )

        # Expected: $0.15 (input) + $0.60 (output) = $0.75
        assert cost == 0.75

    def test_handles_openai_prefix(self):
        """Verify model name parsing handles 'openai:' prefix."""
        tracker = CostTracker()

        cost = tracker.track_completion(
            model="openai:gpt-4o-mini",  # With prefix
            input_tokens=100_000,
            output_tokens=100_000,
            agent_name="TestAgent"
        )

        # Should parse correctly and use gpt-4o-mini pricing
        expected = (100_000 / 1_000_000) * 0.15 + (100_000 / 1_000_000) * 0.60
        assert cost == pytest.approx(expected, rel=1e-6)

    def test_per_agent_tracking(self):
        """Verify per-agent cost attribution."""
        tracker = CostTracker()

        tracker.track_completion(
            model="gpt-4o-mini",
            input_tokens=100_000,
            output_tokens=50_000,
            agent_name="ClassifierAgent"
        )

        tracker.track_completion(
            model="gpt-4o",
            input_tokens=200_000,
            output_tokens=100_000,
            agent_name="DirectorAgent"
        )

        tracker.track_completion(
            model="gpt-4o-mini",
            input_tokens=50_000,
            output_tokens=25_000,
            agent_name="ClassifierAgent"
        )

        # ClassifierAgent should have 2 calls tracked
        assert "ClassifierAgent" in tracker.per_agent_costs
        assert "DirectorAgent" in tracker.per_agent_costs

        # Verify ClassifierAgent cost is sum of both calls
        expected_classifier = (
            (100_000 / 1_000_000) * 0.15 + (50_000 / 1_000_000) * 0.60 +
            (50_000 / 1_000_000) * 0.15 + (25_000 / 1_000_000) * 0.60
        )
        assert tracker.per_agent_costs["ClassifierAgent"] == pytest.approx(expected_classifier, rel=1e-6)

    def test_call_count_tracking(self):
        """Verify call count increments correctly."""
        tracker = CostTracker()

        for i in range(5):
            tracker.track_completion(
                model="gpt-4o-mini",
                input_tokens=1000,
                output_tokens=500,
                agent_name="TestAgent"
            )

        assert tracker.lifetime_usage.call_count == 5
        assert tracker.hourly_usage.call_count == 5
        assert tracker.daily_usage.call_count == 5

    def test_unknown_model_fallback(self):
        """Verify unknown models default to gpt-4o pricing with warning."""
        tracker = CostTracker()

        cost = tracker.track_completion(
            model="gpt-5-ultra",  # Non-existent model
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            agent_name="TestAgent"
        )

        # Should default to gpt-4o pricing
        assert cost == 12.50

    def test_budget_enforcement_hourly(self, monkeypatch):
        """Verify hourly budget limit triggers exception."""
        # Temporarily set environment to development to enable budget enforcement
        monkeypatch.setenv("ENVIRONMENT", "development")
        get_settings.cache_clear()

        tracker = CostTracker()
        tracker.settings.hourly_cost_limit_usd = 1.0  # Set very low limit

        # First call should succeed
        tracker.track_completion(
            model="gpt-4o-mini",
            input_tokens=100_000,
            output_tokens=50_000,
            agent_name="TestAgent"
        )

        # Second call should exceed hourly limit
        with pytest.raises(RuntimeError, match="HOURLY BUDGET EXCEEDED"):
            tracker.track_completion(
                model="gpt-4o",
                input_tokens=1_000_000,
                output_tokens=1_000_000,
                agent_name="TestAgent"
            )

        # Restore test environment
        get_settings.cache_clear()

    def test_budget_enforcement_daily(self, monkeypatch):
        """Verify daily budget limit triggers exception."""
        # Temporarily set environment to development to enable budget enforcement
        monkeypatch.setenv("ENVIRONMENT", "development")
        get_settings.cache_clear()

        tracker = CostTracker()
        tracker.settings.daily_cost_limit_usd = 1.0  # Set very low limit
        tracker.settings.hourly_cost_limit_usd = 100.0  # High hourly to test daily

        with pytest.raises(RuntimeError, match="DAILY BUDGET EXCEEDED"):
            tracker.track_completion(
                model="gpt-4o",
                input_tokens=1_000_000,
                output_tokens=1_000_000,
                agent_name="TestAgent"
            )

        # Restore test environment
        get_settings.cache_clear()

    def test_summary_generation(self):
        """Verify get_summary() returns correct structure."""
        tracker = CostTracker()

        tracker.track_completion(
            model="gpt-4o-mini",
            input_tokens=10_000,
            output_tokens=5_000,
            agent_name="TestAgent"
        )

        summary = tracker.get_summary()

        # Verify structure
        assert "hourly" in summary
        assert "daily" in summary
        assert "lifetime" in summary
        assert "per_agent" in summary

        # Verify hourly data
        assert "cost_usd" in summary["hourly"]
        assert "tokens" in summary["hourly"]
        assert "calls" in summary["hourly"]
        assert "pct_used" in summary["hourly"]

        # Verify values
        assert summary["hourly"]["calls"] == 1
        assert summary["hourly"]["tokens"] == 15_000
        assert summary["per_agent"]["TestAgent"] > 0

    def test_singleton_pattern(self):
        """Verify get_cost_tracker() returns same instance."""
        tracker1 = get_cost_tracker()
        tracker2 = get_cost_tracker()

        assert tracker1 is tracker2

        # Modify one and verify other sees changes
        tracker1.track_completion(
            model="gpt-4o-mini",
            input_tokens=1000,
            output_tokens=500,
            agent_name="TestAgent"
        )

        assert tracker2.lifetime_usage.call_count == 1
