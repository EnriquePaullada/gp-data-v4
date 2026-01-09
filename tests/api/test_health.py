"""
Tests for health and readiness endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock, patch

from src.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_orchestrator():
    """Create mock orchestrator with repositories."""
    orchestrator = MagicMock()
    orchestrator.lead_repo = MagicMock()
    orchestrator.message_repo = MagicMock()
    return orchestrator


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_200(self, client):
        """Health endpoint returns 200 OK."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "gp-data-v4"
        assert "version" in data

    def test_health_includes_version(self, client):
        """Health endpoint includes API version."""
        response = client.get("/health")

        data = response.json()
        assert data["version"] == "1.3.0"


class TestReadinessEndpoint:
    """Tests for /ready endpoint."""

    @pytest.mark.asyncio
    async def test_ready_when_initialized(self, client, mock_orchestrator):
        """Ready endpoint returns 200 when orchestrator is initialized."""
        app.state.orchestrator = mock_orchestrator

        # Mock the db_manager module's client attribute
        mock_client = MagicMock()
        mock_admin = MagicMock()
        mock_admin.command = AsyncMock(return_value={"ok": 1})
        mock_client.admin = mock_admin

        with patch("src.api.routes.health.db_manager") as mock_db:
            mock_db.client = mock_client

            response = client.get("/ready")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ready"
            assert data["mongodb"] == "connected"
            assert data["orchestrator"] == "initialized"

    @pytest.mark.asyncio
    async def test_ready_fails_when_repos_not_initialized(self, client):
        """Ready endpoint returns 503 when repositories not initialized."""
        incomplete_orchestrator = MagicMock()
        incomplete_orchestrator.lead_repo = None
        incomplete_orchestrator.message_repo = None

        app.state.orchestrator = incomplete_orchestrator

        response = client.get("/ready")

        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "not_ready"
        assert "Repositories not initialized" in data["reason"]

    @pytest.mark.asyncio
    async def test_ready_fails_when_mongodb_down(self, client, mock_orchestrator):
        """Ready endpoint returns 503 when MongoDB is not reachable."""
        app.state.orchestrator = mock_orchestrator

        mock_client = MagicMock()
        mock_admin = MagicMock()
        mock_admin.command = AsyncMock(side_effect=Exception("Connection refused"))
        mock_client.admin = mock_admin

        with patch("src.api.routes.health.db_manager") as mock_db:
            mock_db.client = mock_client

            response = client.get("/ready")

            assert response.status_code == 503
            data = response.json()
            assert data["status"] == "not_ready"


class TestRootEndpoint:
    """Tests for / root endpoint."""

    def test_root_returns_api_info(self, client):
        """Root endpoint returns API information."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "GP Data v4 API"
        assert "version" in data
        assert "endpoints" in data

    def test_root_lists_all_endpoints(self, client):
        """Root endpoint lists all available endpoints."""
        response = client.get("/")

        data = response.json()
        endpoints = data["endpoints"]

        assert "health" in endpoints
        assert "ready" in endpoints
        assert "metrics" in endpoints
        assert "twilio_webhook" in endpoints
        assert "queue_metrics" in endpoints
