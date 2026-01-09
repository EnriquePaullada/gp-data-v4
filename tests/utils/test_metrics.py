"""
Tests for Prometheus metrics collection.
"""
import pytest
from src.utils.metrics import (
    MetricsRegistry,
    Counter,
    Gauge,
    Histogram,
    Timer,
    metrics,
)


class TestCounter:
    """Tests for Counter metric type."""

    def test_counter_increment(self):
        """Counter increments correctly."""
        counter = Counter("test_counter", "Test counter")
        counter.inc()
        counter.inc(5)

        values = counter.collect()
        assert len(values) == 1
        assert values[0].value == 6

    def test_counter_with_labels(self):
        """Counter tracks separate values per label combination."""
        counter = Counter("test_counter", "Test counter", ["status"])
        counter.inc(1, status="success")
        counter.inc(2, status="error")
        counter.inc(1, status="success")

        values = counter.collect()
        assert len(values) == 2

        success_value = next(v for v in values if v.labels.get("status") == "success")
        error_value = next(v for v in values if v.labels.get("status") == "error")

        assert success_value.value == 2
        assert error_value.value == 2


class TestGauge:
    """Tests for Gauge metric type."""

    def test_gauge_set(self):
        """Gauge sets value correctly."""
        gauge = Gauge("test_gauge", "Test gauge")
        gauge.set(42)

        values = gauge.collect()
        assert len(values) == 1
        assert values[0].value == 42

    def test_gauge_inc_dec(self):
        """Gauge increments and decrements correctly."""
        gauge = Gauge("test_gauge", "Test gauge")
        gauge.set(10)
        gauge.inc(5)
        gauge.dec(3)

        values = gauge.collect()
        assert values[0].value == 12

    def test_gauge_with_labels(self):
        """Gauge tracks separate values per label combination."""
        gauge = Gauge("test_gauge", "Test gauge", ["region"])
        gauge.set(100, region="us-east")
        gauge.set(200, region="us-west")

        values = gauge.collect()
        assert len(values) == 2


class TestHistogram:
    """Tests for Histogram metric type."""

    def test_histogram_observe(self):
        """Histogram records observations correctly."""
        histogram = Histogram(
            "test_histogram",
            "Test histogram",
            buckets=(0.1, 0.5, 1.0)
        )
        histogram.observe(0.05)
        histogram.observe(0.3)
        histogram.observe(0.8)
        histogram.observe(2.0)

        values = histogram.collect()

        # Find bucket values
        bucket_01 = next(v for v in values if v.labels.get("le") == "0.1")
        bucket_05 = next(v for v in values if v.labels.get("le") == "0.5")
        bucket_10 = next(v for v in values if v.labels.get("le") == "1.0")
        bucket_inf = next(v for v in values if v.labels.get("le") == "+Inf")

        # Cumulative counts
        assert bucket_01.value == 1  # 0.05
        assert bucket_05.value == 2  # 0.05, 0.3
        assert bucket_10.value == 3  # 0.05, 0.3, 0.8
        assert bucket_inf.value == 4  # all

        # Sum and count
        sum_value = next(v for v in values if v.labels.get("_metric") == "sum")
        count_value = next(v for v in values if v.labels.get("_metric") == "count")

        assert sum_value.value == pytest.approx(0.05 + 0.3 + 0.8 + 2.0)
        assert count_value.value == 4


class TestTimer:
    """Tests for Timer context manager."""

    def test_timer_records_duration(self):
        """Timer records duration to histogram."""
        histogram = Histogram("test_duration", "Test duration")

        with Timer(histogram, endpoint="test"):
            pass  # Instant execution

        values = histogram.collect()
        count_value = next(v for v in values if v.labels.get("_metric") == "count")
        assert count_value.value == 1


class TestMetricsRegistry:
    """Tests for MetricsRegistry singleton."""

    def test_registry_is_singleton(self):
        """MetricsRegistry returns same instance."""
        registry1 = MetricsRegistry()
        registry2 = MetricsRegistry()
        assert registry1 is registry2

    def test_registry_has_application_metrics(self):
        """Registry initializes with expected application metrics."""
        registry = MetricsRegistry()

        assert hasattr(registry, 'requests_total')
        assert hasattr(registry, 'queue_pending')
        assert hasattr(registry, 'pipeline_duration')
        assert hasattr(registry, 'tokens_input')
        assert hasattr(registry, 'cost_usd')

    def test_track_agent_tokens(self):
        """track_agent_tokens convenience method works."""
        registry = MetricsRegistry()
        registry.reset()  # Clean state for test

        registry.track_agent_tokens(
            agent="classifier",
            input_tokens=100,
            output_tokens=50,
            input_cost_usd=0.001,
            output_cost_usd=0.002
        )

        # Verify tokens tracked
        input_values = registry.tokens_input.collect()
        classifier_input = next(
            (v for v in input_values if v.labels.get("agent") == "classifier"),
            None
        )
        assert classifier_input is not None
        assert classifier_input.value == 100

        # Verify cost tracked
        cost_values = registry.cost_usd.collect()
        classifier_cost = next(
            (v for v in cost_values if v.labels.get("agent") == "classifier"),
            None
        )
        assert classifier_cost is not None
        assert classifier_cost.value == pytest.approx(0.003)

    def test_export_prometheus_format(self):
        """Export generates valid Prometheus text format."""
        registry = MetricsRegistry()
        registry.reset()

        # Add some test data
        registry.requests_total.inc(status="success")
        registry.queue_pending.set(5)

        output = registry.export()

        # Check format
        assert "# HELP gp_requests_total" in output
        assert "# TYPE gp_requests_total counter" in output
        assert 'gp_requests_total{status="success"} 1' in output

        assert "# HELP gp_queue_pending" in output
        assert "# TYPE gp_queue_pending gauge" in output
        assert "gp_queue_pending 5" in output

    def test_reset_clears_metrics(self):
        """Reset clears all metric values."""
        registry = MetricsRegistry()
        registry.requests_total.inc(status="test")

        registry.reset()

        values = registry.requests_total.collect()
        assert len(values) == 0


class TestGlobalMetrics:
    """Tests for global metrics instance."""

    def test_global_metrics_accessible(self):
        """Global metrics instance is accessible."""
        assert metrics is not None
        assert isinstance(metrics, MetricsRegistry)

    def test_global_metrics_has_all_expected_metrics(self):
        """Global metrics has all expected application metrics."""
        expected = [
            'requests_total',
            'request_duration',
            'queue_pending',
            'queue_processing',
            'queue_failed',
            'queue_completed',
            'pipeline_duration',
            'agent_calls',
            'agent_errors',
            'agent_duration',
            'tokens_input',
            'tokens_output',
            'tokens_total',
            'cost_usd',
            'cost_input_usd',
            'cost_output_usd',
            'hourly_cost_usd',
            'daily_cost_usd',
            'rate_limit_hits',
            'bans_total',
            'active_bans',
            'security_blocks',
            'pii_redactions',
        ]

        for metric_name in expected:
            assert hasattr(metrics, metric_name), f"Missing metric: {metric_name}"
