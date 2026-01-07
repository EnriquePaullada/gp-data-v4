"""
Tests for FastAPI application endpoints.
"""
import pytest
import os
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock

from src.api.main import app
from src.models.lead import Lead, SalesStage
from src.core.conversation_orchestrator import OrchestrationResult, SecurityException
from src.models.classifier_response import ClassifierResponse, Intent
from src.models.director_response import DirectorResponse, StrategicAction, MessageStrategy
from src.models.executor_response import ExecutorResponse, OutboundMessage
from src.repositories import db_manager


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


@pytest.fixture
def mock_queue():
    """Create mock message queue."""
    queue = MagicMock()
    queue.enqueue = AsyncMock(return_value="msg-123")
    queue.get_metrics = AsyncMock()
    return queue


class TestHealthEndpoints:
    """Test health check and readiness endpoints."""

    def test_health_check(self, client):
        """Test basic health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "gp-data-v4"
        assert "version" in data

    @pytest.mark.asyncio
    async def test_readiness_check_when_ready(self, client, mock_orchestrator):
        """Test readiness check when service is ready."""
        # Mock app state
        app.state.orchestrator = mock_orchestrator

        # Mock MongoDB ping - patch the property method directly
        mock_client = MagicMock()
        mock_client.admin.command = AsyncMock(return_value={"ok": 1})

        with patch.object(type(db_manager), "client", property(lambda self: mock_client)):
            response = client.get("/ready")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ready"
            assert data["mongodb"] == "connected"

    @pytest.mark.asyncio
    async def test_readiness_check_when_not_ready(self, client):
        """Test readiness check when repositories not initialized."""
        mock_orch = MagicMock()
        mock_orch.lead_repo = None  # Not initialized
        mock_orch.message_repo = None

        app.state.orchestrator = mock_orch

        response = client.get("/ready")

        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "not_ready"

    def test_root_endpoint(self, client):
        """Test root endpoint returns API information."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "GP Data v4 API"
        assert "endpoints" in data


class TestTwilioWebhook:
    """Test Twilio webhook endpoint."""

    @pytest.fixture
    def valid_twilio_payload(self):
        """Create valid Twilio webhook payload."""
        return {
            "MessageSid": "SM1234567890abcdef",
            "AccountSid": "AC1234567890abcdef",
            "From": "whatsapp:+5215538899800",
            "To": "whatsapp:+14155238886",
            "Body": "I need pricing for 20 users",
            "NumMedia": "0",
            "ProfileName": "Carlos Rodriguez",
            "WaId": "5215538899800"
        }

    @pytest.fixture
    def mock_orchestration_result(self):
        """Create mock orchestration result."""
        from src.models.intelligence import Sentiment, BANTDimension

        lead = Lead(
            lead_id="+5215538899800",
            full_name="Carlos Rodriguez",
            current_stage=SalesStage.DISCOVERY
        )

        return OrchestrationResult(
            outbound_message="Thank you for your interest! I'd be happy to discuss pricing for your team of 20 users.",
            classification=ClassifierResponse(
                intent=Intent.PRICING,
                intent_confidence=0.95,
                topic="pricing",
                topic_confidence=0.90,
                sentiment=Sentiment.POSITIVE,
                urgency="medium",
                urgency_confidence=0.85,
                language="english",
                engagement_level="high",
                requires_human_escalation=False,
                reasoning="Lead asking about pricing",
                new_signals=[]
            ),
            strategy=DirectorResponse(
                action=StrategicAction.QUALIFY,
                message_strategy=MessageStrategy(
                    tone="professional",
                    language="english",
                    empathy_points=["Understanding team needs"],
                    key_points=["pricing", "team size"],
                    conversational_goal="Qualify budget"
                ),
                focus_dimension=BANTDimension.BUDGET,
                strategic_reasoning="Qualify budget"
            ),
            execution=ExecutorResponse(
                message=OutboundMessage(
                    content="Thank you for your interest! I'd be happy to discuss pricing for your team of 20 users.",
                    persona_reasoning="Professional and helpful tone to build trust"
                ),
                agreement_level=0.95,
                execution_summary="Responded to pricing inquiry"
            ),
            total_duration_ms=2500.0,
            lead_updated=lead
        )

    @pytest.mark.asyncio
    async def test_twilio_webhook_success(self, client, valid_twilio_payload, mock_queue):
        """Test successful webhook enqueueing."""
        # Mock the queue
        app.state.queue = mock_queue

        response = client.post("/webhooks/twilio", data=valid_twilio_payload)

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

        # Verify JSON response
        data = response.json()
        assert data["status"] == "queued"
        assert data["message_id"] == "msg-123"
        assert data["phone"] == "+5215538899800"

        # Verify message was enqueued
        mock_queue.enqueue.assert_called_once()

        # Verify the queued message has correct data
        call_args = mock_queue.enqueue.call_args[0][0]
        assert call_args.phone == "+5215538899800"
        assert call_args.body == "I need pricing for 20 users"
        assert call_args.message_sid == "SM1234567890abcdef"
        assert call_args.profile_name == "Carlos Rodriguez"

    @pytest.mark.asyncio
    async def test_twilio_webhook_with_security_exception(self, client, valid_twilio_payload, mock_queue):
        """Test webhook enqueueing (security validation happens in worker)."""
        # Note: Security validation now happens in the worker, not in the webhook endpoint
        # The webhook just enqueues the message
        app.state.queue = mock_queue

        response = client.post("/webhooks/twilio", data=valid_twilio_payload)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "queued"

        # Message should still be enqueued - security check happens in worker
        mock_queue.enqueue.assert_called_once()

    @pytest.mark.asyncio
    async def test_twilio_webhook_with_processing_error(self, client, valid_twilio_payload, mock_queue):
        """Test webhook when enqueueing fails."""
        # Mock queue.enqueue to raise exception
        mock_queue.enqueue = AsyncMock(side_effect=Exception("Queue full"))

        app.state.queue = mock_queue
        with patch("src.api.main.twilio_service.send_whatsapp_message", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = "SM123456789"

            response = client.post("/webhooks/twilio", data=valid_twilio_payload)

            assert response.status_code == 500  # Queue error returns 500
            assert "application/json" in response.headers["content-type"]

            # Verify JSON response
            data = response.json()
            assert data["status"] == "error"
            assert data["phone"] == "+5215538899800"

            # Verify fallback message was sent via Twilio
            mock_send.assert_called_once_with(
                to_number="+5215538899800",
                message="I'm experiencing technical difficulties. Please try again in a moment."
            )

    def test_twilio_webhook_missing_required_fields(self, client):
        """Test webhook with missing required fields."""
        incomplete_payload = {
            "MessageSid": "SM123",
            # Missing From, To, Body, etc.
        }

        response = client.post("/webhooks/twilio", data=incomplete_payload)

        # FastAPI will return 422 for validation errors
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_twilio_webhook_phone_extraction(self, client, valid_twilio_payload, mock_queue):
        """Test that phone number is correctly extracted from WhatsApp format."""
        app.state.queue = mock_queue

        response = client.post("/webhooks/twilio", data=valid_twilio_payload)

        assert response.status_code == 200

        # Verify phone was cleaned (removed "whatsapp:" prefix)
        call_args = mock_queue.enqueue.call_args[0][0]
        assert call_args.phone == "+5215538899800"
        assert "whatsapp:" not in call_args.phone

    @pytest.mark.asyncio
    async def test_twilio_webhook_no_profile_name(self, client, mock_queue):
        """Test webhook when ProfileName is not provided."""
        payload = {
            "MessageSid": "SM123",
            "AccountSid": "AC123",
            "From": "whatsapp:+1234567890",
            "To": "whatsapp:+14155238886",
            "Body": "Hello",
            "NumMedia": "0",
            # No ProfileName
        }

        app.state.queue = mock_queue

        response = client.post("/webhooks/twilio", data=payload)

        assert response.status_code == 200

        # Should use phone number as fallback name
        call_args = mock_queue.enqueue.call_args[0][0]
        assert call_args.profile_name == "+1234567890"


class TestTwilioModels:
    """Test Twilio Pydantic models."""

    def test_twilio_payload_phone_extraction(self):
        """Test phone extraction from WhatsApp format."""
        from src.api.models.twilio import TwilioWebhookPayload

        payload = TwilioWebhookPayload(
            MessageSid="SM123",
            AccountSid="AC123",
            From="whatsapp:+5215538899800",
            To="whatsapp:+14155238886",
            Body="Test message"
        )

        assert payload.get_clean_phone() == "+5215538899800"

    def test_twilio_payload_profile_name_fallback(self):
        """Test profile name fallback to phone."""
        from src.api.models.twilio import TwilioWebhookPayload

        payload = TwilioWebhookPayload(
            MessageSid="SM123",
            AccountSid="AC123",
            From="whatsapp:+1234567890",
            To="whatsapp:+14155238886",
            Body="Test"
        )

        assert payload.get_profile_name() == "+1234567890"

        # With ProfileName provided
        payload.ProfileName = "John Doe"
        assert payload.get_profile_name() == "John Doe"
