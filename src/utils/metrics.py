"""
Prometheus Metrics Collector

Lightweight metrics collection for observability without external dependencies.
Generates Prometheus text exposition format (text/plain; version=0.0.4).
"""
import time
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class MetricType(str, Enum):
    """Prometheus metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


@dataclass
class MetricValue:
    """Single metric value with optional labels."""
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp_ms: Optional[int] = None


class Counter:
    """
    Prometheus Counter metric.

    A counter is a cumulative metric that only goes up.
    Used for: request counts, error counts, token usage, costs, etc.
    """

    def __init__(self, name: str, description: str, labels: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._values: Dict[tuple, float] = {}
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        """Increment counter by amount."""
        key = self._label_key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def _label_key(self, labels: Dict[str, str]) -> tuple:
        """Create hashable key from labels."""
        return tuple(sorted(labels.items()))

    def collect(self) -> List[MetricValue]:
        """Collect all metric values."""
        with self._lock:
            return [
                MetricValue(value=v, labels=dict(k))
                for k, v in self._values.items()
            ]


class Gauge:
    """
    Prometheus Gauge metric.

    A gauge can go up and down.
    Used for: queue depth, active connections, current costs, etc.
    """

    def __init__(self, name: str, description: str, labels: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._values: Dict[tuple, float] = {}
        self._lock = threading.Lock()

    def set(self, value: float, **labels: str) -> None:
        """Set gauge to value."""
        key = self._label_key(labels)
        with self._lock:
            self._values[key] = value

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        """Increment gauge by amount."""
        key = self._label_key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def dec(self, amount: float = 1.0, **labels: str) -> None:
        """Decrement gauge by amount."""
        key = self._label_key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) - amount

    def _label_key(self, labels: Dict[str, str]) -> tuple:
        """Create hashable key from labels."""
        return tuple(sorted(labels.items()))

    def collect(self) -> List[MetricValue]:
        """Collect all metric values."""
        with self._lock:
            return [
                MetricValue(value=v, labels=dict(k))
                for k, v in self._values.items()
            ]


class Histogram:
    """
    Prometheus Histogram metric.

    Samples observations and counts them in buckets.
    Used for: request duration, response sizes, etc.
    """

    # Default buckets suitable for HTTP request latencies (in seconds)
    DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

    def __init__(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
        buckets: Optional[tuple] = None
    ):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self.buckets = buckets or self.DEFAULT_BUCKETS
        self._values: Dict[tuple, Dict] = {}
        self._lock = threading.Lock()

    def observe(self, value: float, **labels: str) -> None:
        """Record an observation."""
        key = self._label_key(labels)
        with self._lock:
            if key not in self._values:
                self._values[key] = {
                    "buckets": {b: 0 for b in self.buckets},
                    "sum": 0.0,
                    "count": 0
                }

            data = self._values[key]
            data["sum"] += value
            data["count"] += 1

            for bucket in self.buckets:
                if value <= bucket:
                    data["buckets"][bucket] += 1

    def _label_key(self, labels: Dict[str, str]) -> tuple:
        """Create hashable key from labels."""
        return tuple(sorted(labels.items()))

    def collect(self) -> List[MetricValue]:
        """Collect all metric values including buckets, sum, and count."""
        result = []
        with self._lock:
            for key, data in self._values.items():
                base_labels = dict(key)

                # Bucket values (already cumulative from observe())
                for bucket in sorted(self.buckets):
                    result.append(MetricValue(
                        value=data["buckets"][bucket],
                        labels={**base_labels, "le": str(bucket)}
                    ))

                # +Inf bucket
                result.append(MetricValue(
                    value=data["count"],
                    labels={**base_labels, "le": "+Inf"}
                ))

                # Sum and count
                result.append(MetricValue(
                    value=data["sum"],
                    labels={**base_labels, "_metric": "sum"}
                ))
                result.append(MetricValue(
                    value=data["count"],
                    labels={**base_labels, "_metric": "count"}
                ))

        return result


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, histogram: Histogram, **labels: str):
        self.histogram = histogram
        self.labels = labels
        self.start_time: Optional[float] = None

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        if self.start_time is not None:
            duration = time.perf_counter() - self.start_time
            self.histogram.observe(duration, **self.labels)


class MetricsRegistry:
    """
    Central registry for all application metrics.

    Provides singleton access and Prometheus text format export.
    """

    _instance: Optional["MetricsRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "MetricsRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._metrics: Dict[str, Counter | Gauge | Histogram] = {}
        self._initialized = True

        # Initialize application metrics
        self._setup_metrics()

    def _setup_metrics(self) -> None:
        """Initialize all application metrics."""

        # ============================================
        # REQUEST METRICS
        # ============================================
        self.requests_total = self.counter(
            "gp_requests_total",
            "Total HTTP requests by status",
            ["status"]
        )

        self.request_duration = self.histogram(
            "gp_request_duration_seconds",
            "HTTP request duration in seconds",
            ["endpoint"]
        )

        # ============================================
        # QUEUE METRICS
        # ============================================
        self.queue_pending = self.gauge(
            "gp_queue_pending",
            "Number of messages pending in queue"
        )

        self.queue_processing = self.gauge(
            "gp_queue_processing",
            "Number of messages currently being processed"
        )

        self.queue_failed = self.gauge(
            "gp_queue_failed",
            "Number of messages in dead letter queue"
        )

        self.queue_completed = self.counter(
            "gp_queue_completed_total",
            "Total messages processed successfully"
        )

        # ============================================
        # PIPELINE METRICS
        # ============================================
        self.pipeline_duration = self.histogram(
            "gp_pipeline_duration_seconds",
            "3-agent pipeline processing duration",
            buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
        )

        self.agent_calls = self.counter(
            "gp_agent_calls_total",
            "Total agent invocations by agent type",
            ["agent"]
        )

        self.agent_errors = self.counter(
            "gp_agent_errors_total",
            "Total agent errors by agent type",
            ["agent"]
        )

        self.agent_duration = self.histogram(
            "gp_agent_duration_seconds",
            "Individual agent processing duration",
            ["agent"],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
        )

        # ============================================
        # TOKEN USAGE METRICS
        # ============================================
        self.tokens_input = self.counter(
            "gp_tokens_input_total",
            "Total input tokens consumed by agent",
            ["agent"]
        )

        self.tokens_output = self.counter(
            "gp_tokens_output_total",
            "Total output tokens generated by agent",
            ["agent"]
        )

        self.tokens_total = self.counter(
            "gp_tokens_total",
            "Total tokens (input + output) by agent and type",
            ["agent", "type"]
        )

        # ============================================
        # COST METRICS (USD)
        # ============================================
        self.cost_usd = self.counter(
            "gp_cost_usd_total",
            "Total cost in USD by agent",
            ["agent"]
        )

        self.cost_input_usd = self.counter(
            "gp_cost_input_usd_total",
            "Total input token cost in USD by agent",
            ["agent"]
        )

        self.cost_output_usd = self.counter(
            "gp_cost_output_usd_total",
            "Total output token cost in USD by agent",
            ["agent"]
        )

        self.hourly_cost_usd = self.gauge(
            "gp_hourly_cost_usd",
            "Cost in current hour window (USD)"
        )

        self.daily_cost_usd = self.gauge(
            "gp_daily_cost_usd",
            "Cost in current day window (USD)"
        )

        # ============================================
        # RATE LIMITING METRICS
        # ============================================
        self.rate_limit_hits = self.counter(
            "gp_rate_limit_hits_total",
            "Total rate limit enforcement events"
        )

        self.bans_total = self.counter(
            "gp_bans_total",
            "Total lead bans issued",
            ["reason"]
        )

        self.active_bans = self.gauge(
            "gp_active_bans",
            "Number of currently active bans"
        )

        # ============================================
        # SECURITY METRICS
        # ============================================
        self.security_blocks = self.counter(
            "gp_security_blocks_total",
            "Messages blocked by security validator",
            ["reason"]
        )

        self.pii_redactions = self.counter(
            "gp_pii_redactions_total",
            "PII redaction events by type",
            ["type"]
        )

    def counter(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None
    ) -> Counter:
        """Create and register a counter."""
        metric = Counter(name, description, labels)
        self._metrics[name] = metric
        return metric

    def gauge(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None
    ) -> Gauge:
        """Create and register a gauge."""
        metric = Gauge(name, description, labels)
        self._metrics[name] = metric
        return metric

    def histogram(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
        buckets: Optional[tuple] = None
    ) -> Histogram:
        """Create and register a histogram."""
        metric = Histogram(name, description, labels, buckets)
        self._metrics[name] = metric
        return metric

    def track_agent_tokens(
        self,
        agent: str,
        input_tokens: int,
        output_tokens: int,
        input_cost_usd: float,
        output_cost_usd: float
    ) -> None:
        """
        Convenience method to track all token and cost metrics for an agent call.

        Args:
            agent: Agent name (classifier, director, executor)
            input_tokens: Number of input/prompt tokens
            output_tokens: Number of output/completion tokens
            input_cost_usd: Cost of input tokens in USD
            output_cost_usd: Cost of output tokens in USD
        """
        # Token counts
        self.tokens_input.inc(input_tokens, agent=agent)
        self.tokens_output.inc(output_tokens, agent=agent)
        self.tokens_total.inc(input_tokens, agent=agent, type="input")
        self.tokens_total.inc(output_tokens, agent=agent, type="output")

        # Costs
        total_cost = input_cost_usd + output_cost_usd
        self.cost_input_usd.inc(input_cost_usd, agent=agent)
        self.cost_output_usd.inc(output_cost_usd, agent=agent)
        self.cost_usd.inc(total_cost, agent=agent)

    def export(self) -> str:
        """
        Export all metrics in Prometheus text exposition format.

        Format specification:
        https://prometheus.io/docs/instrumenting/exposition_formats/
        """
        lines = []

        for name, metric in self._metrics.items():
            # Add HELP and TYPE
            lines.append(f"# HELP {name} {metric.description}")

            if isinstance(metric, Counter):
                lines.append(f"# TYPE {name} counter")
            elif isinstance(metric, Gauge):
                lines.append(f"# TYPE {name} gauge")
            elif isinstance(metric, Histogram):
                lines.append(f"# TYPE {name} histogram")

            # Add metric values
            for mv in metric.collect():
                if isinstance(metric, Histogram):
                    # Handle histogram specially
                    if "_metric" in mv.labels:
                        suffix = "_" + mv.labels.pop("_metric")
                        metric_name = f"{name}{suffix}"
                    elif "le" in mv.labels:
                        metric_name = f"{name}_bucket"
                    else:
                        metric_name = name
                else:
                    metric_name = name

                label_str = self._format_labels(mv.labels)
                lines.append(f"{metric_name}{label_str} {mv.value}")

            lines.append("")  # Empty line between metrics

        return "\n".join(lines)

    def _format_labels(self, labels: Dict[str, str]) -> str:
        """Format labels as Prometheus label string."""
        if not labels:
            return ""

        parts = [f'{k}="{v}"' for k, v in sorted(labels.items())]
        return "{" + ",".join(parts) + "}"

    def reset(self) -> None:
        """Reset all metrics. Useful for testing."""
        self._metrics.clear()
        self._setup_metrics()


# Global metrics instance
metrics = MetricsRegistry()
