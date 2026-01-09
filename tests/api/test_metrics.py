"""
Tests for metrics endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock

from src.api.main import app
from src.message_queue import QueueMetrics


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_queue_with_metrics():
    """Create mock queue with realistic metrics."""
    queue = MagicMock()
    queue.get_metrics = AsyncMock(return_value=QueueMetrics(
        pending=5,
        processing=2,
        completed=100,
        failed=3,
        dead_letter=1,
        avg_processing_time_ms=150.5,
        error_rate=0.03
    ))
    return queue


class TestPrometheusMetricsEndpoint:
    """Tests for /metrics Prometheus endpoint."""

    def test_returns_prometheus_format(self, client, mock_queue_with_metrics):
        """Test /metrics returns Prometheus text format."""
        app.state.queue = mock_queue_with_metrics

        response = client.get("/metrics")

        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]

        content = response.text
        assert "# HELP" in content
        assert "# TYPE" in content

    def test_includes_queue_metrics(self, client, mock_queue_with_metrics):
        """Test /metrics includes queue gauge metrics."""
        app.state.queue = mock_queue_with_metrics

        response = client.get("/metrics")

        content = response.text
        assert "gp_queue_pending" in content
        assert "gp_queue_processing" in content
        assert "gp_queue_failed" in content

    def test_updates_queue_gauges_from_state(self, client, mock_queue_with_metrics):
        """Test /metrics updates queue gauges with current values."""
        app.state.queue = mock_queue_with_metrics

        response = client.get("/metrics")

        content = response.text
        assert "gp_queue_pending 5" in content
        assert "gp_queue_processing 2" in content
        assert "gp_queue_failed 3" in content

    def test_includes_counter_metrics(self, client, mock_queue_with_metrics):
        """Test /metrics includes counter metric types."""
        app.state.queue = mock_queue_with_metrics

        response = client.get("/metrics")

        content = response.text
        assert "# TYPE gp_requests_total counter" in content
        assert "# TYPE gp_queue_completed_total counter" in content
        assert "# TYPE gp_agent_calls_total counter" in content

    def test_includes_histogram_metrics(self, client, mock_queue_with_metrics):
        """Test /metrics includes histogram metric types."""
        app.state.queue = mock_queue_with_metrics

        response = client.get("/metrics")

        content = response.text
        assert "# TYPE gp_pipeline_duration_seconds histogram" in content
        assert "# TYPE gp_request_duration_seconds histogram" in content
        assert "# TYPE gp_agent_duration_seconds histogram" in content

    def test_includes_token_metrics(self, client, mock_queue_with_metrics):
        """Test /metrics includes token tracking metrics."""
        app.state.queue = mock_queue_with_metrics

        response = client.get("/metrics")

        content = response.text
        assert "# HELP gp_tokens_input_total" in content
        assert "# HELP gp_tokens_output_total" in content
        assert "# TYPE gp_tokens_input_total counter" in content

    def test_includes_cost_metrics(self, client, mock_queue_with_metrics):
        """Test /metrics includes cost tracking metrics."""
        app.state.queue = mock_queue_with_metrics

        response = client.get("/metrics")

        content = response.text
        assert "# HELP gp_cost_usd_total" in content
        assert "# HELP gp_hourly_cost_usd" in content
        assert "# HELP gp_daily_cost_usd" in content

    def test_includes_rate_limiting_metrics(self, client, mock_queue_with_metrics):
        """Test /metrics includes rate limiting metrics."""
        app.state.queue = mock_queue_with_metrics

        response = client.get("/metrics")

        content = response.text
        assert "# HELP gp_rate_limit_hits_total" in content
        assert "# HELP gp_bans_total" in content
        assert "# HELP gp_active_bans" in content

    def test_includes_security_metrics(self, client, mock_queue_with_metrics):
        """Test /metrics includes security metrics."""
        app.state.queue = mock_queue_with_metrics

        response = client.get("/metrics")

        content = response.text
        assert "# HELP gp_security_blocks_total" in content
        assert "# HELP gp_pii_redactions_total" in content

    def test_handles_queue_error_gracefully(self, client):
        """Test /metrics handles queue error gracefully."""
        mock_queue = MagicMock()
        mock_queue.get_metrics = AsyncMock(side_effect=Exception("Queue unavailable"))
        app.state.queue = mock_queue

        response = client.get("/metrics")

        assert response.status_code == 500
        assert "Error exporting metrics" in response.text


class TestQueueMetricsEndpoint:
    """Tests for /metrics/queue JSON endpoint."""

    def test_returns_json_format(self, client, mock_queue_with_metrics):
        """Test /metrics/queue returns JSON format."""
        app.state.queue = mock_queue_with_metrics

        response = client.get("/metrics/queue")

        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]

        data = response.json()
        assert data["status"] == "ok"
        assert "metrics" in data

    def test_includes_all_queue_fields(self, client, mock_queue_with_metrics):
        """Test /metrics/queue includes all expected fields."""
        app.state.queue = mock_queue_with_metrics

        response = client.get("/metrics/queue")
        data = response.json()

        metrics = data["metrics"]
        assert metrics["pending"] == 5
        assert metrics["processing"] == 2
        assert metrics["completed"] == 100
        assert metrics["failed"] == 3
        assert metrics["dead_letter"] == 1
        assert metrics["avg_processing_time_ms"] == 150.5
        assert metrics["error_rate"] == 0.03

    def test_handles_queue_error_gracefully(self, client):
        """Test /metrics/queue handles queue error gracefully."""
        mock_queue = MagicMock()
        mock_queue.get_metrics = AsyncMock(side_effect=Exception("Queue unavailable"))
        app.state.queue = mock_queue

        response = client.get("/metrics/queue")

        assert response.status_code == 500
        data = response.json()
        assert data["status"] == "error"
        assert "Queue unavailable" in data["error"]
