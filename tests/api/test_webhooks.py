"""
Tests for Twilio webhook endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock

from src.api.main import app
from src.config import settings


@pytest.fixture(autouse=True)
def disable_signature_validation(monkeypatch):
    """Disable Twilio signature validation for all tests."""
    monkeypatch.setattr(settings, "twilio_validate_signature", False)


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_message_buffer():
    """Create mock message buffer."""
    buffer = MagicMock()
    buffer.add = AsyncMock()
    buffer.flush_all = AsyncMock()
    buffer.get_buffer_stats = AsyncMock(return_value={
        "pending_leads": 0,
        "total_buffered_messages": 0
    })
    return buffer


@pytest.fixture
def mock_rate_limiter():
    """Create mock rate limiter that allows all requests."""
    from datetime import datetime, timezone

    rate_limiter = MagicMock()
    rate_limiter.is_banned = AsyncMock(return_value=False)
    rate_limiter.check_rate_limit = AsyncMock(return_value=MagicMock(
        allowed=True,
        remaining=9,
        reset_at=datetime.now(timezone.utc),
        retry_after=0,
        reason=None
    ))
    rate_limiter.detect_spike = AsyncMock(return_value=False)
    return rate_limiter


@pytest.fixture
def valid_twilio_payload():
    """Create valid Twilio webhook payload."""
    return {
        "MessageSid": "SM1234567890abcdef",
        "AccountSid": "AC1234567890abcdef",
        "From": "whatsapp:+5215538899800",
        "To": "whatsapp:+14155238886",
        "Body": "I need pricing for 20 users",
        "NumMedia": "0",
        "ProfileName": "Carlos Rodriguez",
    }


class TestTwilioWebhook:
    """Tests for /webhooks/twilio endpoint."""

    @pytest.mark.asyncio
    async def test_webhook_buffers_message(
        self, client, valid_twilio_payload, mock_message_buffer, mock_rate_limiter
    ):
        """Test webhook successfully buffers message."""
        app.state.message_buffer = mock_message_buffer
        app.state.rate_limiter = mock_rate_limiter

        response = client.post("/webhooks/twilio", data=valid_twilio_payload)

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

        data = response.json()
        assert data["status"] == "buffered"
        assert data["message_sid"] == "SM1234567890abcdef"
        assert data["phone"] == "+5215538899800"

        mock_message_buffer.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_webhook_extracts_phone_correctly(
        self, client, valid_twilio_payload, mock_message_buffer, mock_rate_limiter
    ):
        """Test that phone number is correctly extracted from WhatsApp format."""
        app.state.message_buffer = mock_message_buffer
        app.state.rate_limiter = mock_rate_limiter

        response = client.post("/webhooks/twilio", data=valid_twilio_payload)

        assert response.status_code == 200

        call_kwargs = mock_message_buffer.add.call_args[1]
        assert call_kwargs["phone"] == "+5215538899800"
        assert "whatsapp:" not in call_kwargs["phone"]

    @pytest.mark.asyncio
    async def test_webhook_uses_profile_name(
        self, client, valid_twilio_payload, mock_message_buffer, mock_rate_limiter
    ):
        """Test webhook uses ProfileName from payload."""
        app.state.message_buffer = mock_message_buffer
        app.state.rate_limiter = mock_rate_limiter

        response = client.post("/webhooks/twilio", data=valid_twilio_payload)

        assert response.status_code == 200

        call_kwargs = mock_message_buffer.add.call_args[1]
        assert call_kwargs["profile_name"] == "Carlos Rodriguez"

    @pytest.mark.asyncio
    async def test_webhook_falls_back_to_phone_for_name(
        self, client, mock_message_buffer, mock_rate_limiter
    ):
        """Test webhook uses phone number when ProfileName not provided."""
        payload = {
            "MessageSid": "SM123",
            "AccountSid": "AC123",
            "From": "whatsapp:+1234567890",
            "To": "whatsapp:+14155238886",
            "Body": "Hello",
            "NumMedia": "0",
        }

        app.state.message_buffer = mock_message_buffer
        app.state.rate_limiter = mock_rate_limiter

        response = client.post("/webhooks/twilio", data=payload)

        assert response.status_code == 200

        call_kwargs = mock_message_buffer.add.call_args[1]
        assert call_kwargs["profile_name"] == "+1234567890"

    def test_webhook_rejects_missing_required_fields(self, client):
        """Test webhook returns 422 for missing required fields."""
        incomplete_payload = {
            "MessageSid": "SM123",
        }

        response = client.post("/webhooks/twilio", data=incomplete_payload)

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_webhook_handles_buffer_error(
        self, client, valid_twilio_payload, mock_message_buffer, mock_rate_limiter
    ):
        """Test webhook handles buffer error gracefully."""
        mock_message_buffer.add = AsyncMock(side_effect=Exception("Buffer error"))
        app.state.message_buffer = mock_message_buffer
        app.state.rate_limiter = mock_rate_limiter

        with patch(
            "src.api.routes.webhooks.twilio_service.send_whatsapp_message",
            new_callable=AsyncMock
        ) as mock_send:
            mock_send.return_value = "SM123456789"

            response = client.post("/webhooks/twilio", data=valid_twilio_payload)

            assert response.status_code == 500
            data = response.json()
            assert data["status"] == "error"

            mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_webhook_includes_rate_limit_headers(
        self, client, valid_twilio_payload, mock_message_buffer, mock_rate_limiter
    ):
        """Test webhook includes rate limit headers in response."""
        app.state.message_buffer = mock_message_buffer
        app.state.rate_limiter = mock_rate_limiter

        response = client.post("/webhooks/twilio", data=valid_twilio_payload)

        assert response.status_code == 200
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers


class TestRateLimiting:
    """Tests for rate limiting in webhook endpoint."""

    @pytest.mark.asyncio
    async def test_webhook_rejects_rate_limited_request(
        self, client, valid_twilio_payload, mock_message_buffer
    ):
        """Test webhook returns 429 when rate limited."""
        from datetime import datetime, timezone

        rate_limiter = MagicMock()
        rate_limiter.is_banned = AsyncMock(return_value=False)
        rate_limiter.check_rate_limit = AsyncMock(return_value=MagicMock(
            allowed=False,
            remaining=0,
            reset_at=datetime.now(timezone.utc),
            retry_after=60,
            reason="Rate limit exceeded"
        ))

        app.state.message_buffer = mock_message_buffer
        app.state.rate_limiter = rate_limiter

        response = client.post("/webhooks/twilio", data=valid_twilio_payload)

        assert response.status_code == 429
        data = response.json()
        assert data["status"] == "rate_limited"

    @pytest.mark.asyncio
    async def test_webhook_rejects_banned_lead(
        self, client, valid_twilio_payload, mock_message_buffer
    ):
        """Test webhook returns 429 for banned leads."""
        from datetime import datetime, timezone

        rate_limiter = MagicMock()
        rate_limiter.is_banned = AsyncMock(return_value=True)
        rate_limiter.get_ban_info = AsyncMock(return_value=(
            datetime.now(timezone.utc),
            "Abuse detected"
        ))

        app.state.message_buffer = mock_message_buffer
        app.state.rate_limiter = rate_limiter

        with patch(
            "src.api.routes.webhooks.twilio_service.send_whatsapp_message",
            new_callable=AsyncMock
        ):
            response = client.post("/webhooks/twilio", data=valid_twilio_payload)

            assert response.status_code == 429
            data = response.json()
            assert data["status"] == "banned"


class TestSignatureValidation:
    """Tests for Twilio signature validation."""

    @pytest.fixture
    def test_payload(self):
        """Create test Twilio webhook payload."""
        return {
            "MessageSid": "SM1234567890abcdef",
            "AccountSid": "AC1234567890abcdef",
            "From": "whatsapp:+5215538899800",
            "To": "whatsapp:+14155238886",
            "Body": "I need pricing for 20 users",
            "NumMedia": "0",
            "ProfileName": "Carlos Rodriguez",
        }

    @pytest.mark.asyncio
    async def test_accepts_valid_signature(
        self, client, test_payload, mock_message_buffer, mock_rate_limiter, monkeypatch
    ):
        """Test webhook accepts request with valid signature."""
        from src.utils.twilio_signature import TwilioSignatureValidator

        monkeypatch.setattr(settings, "twilio_validate_signature", True)
        monkeypatch.setattr(settings, "twilio_auth_token", "test_token")

        validator = TwilioSignatureValidator("test_token")
        url = "http://testserver/webhooks/twilio"
        params = {k: str(v) for k, v in test_payload.items()}
        valid_signature = validator.compute_signature(url, params)

        app.state.message_buffer = mock_message_buffer
        app.state.rate_limiter = mock_rate_limiter

        response = client.post(
            "/webhooks/twilio",
            data=test_payload,
            headers={"X-Twilio-Signature": valid_signature}
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_rejects_invalid_signature(
        self, client, test_payload, monkeypatch
    ):
        """Test webhook rejects request with invalid signature."""
        monkeypatch.setattr(settings, "twilio_validate_signature", True)
        monkeypatch.setattr(settings, "twilio_auth_token", "test_token")

        response = client.post(
            "/webhooks/twilio",
            data=test_payload,
            headers={"X-Twilio-Signature": "invalid_signature"}
        )

        assert response.status_code == 401
        data = response.json()
        assert data["status"] == "unauthorized"

    @pytest.mark.asyncio
    async def test_rejects_missing_signature(
        self, client, test_payload, monkeypatch
    ):
        """Test webhook rejects request without signature header."""
        monkeypatch.setattr(settings, "twilio_validate_signature", True)
        monkeypatch.setattr(settings, "twilio_auth_token", "test_token")

        response = client.post("/webhooks/twilio", data=test_payload)

        assert response.status_code == 401
        data = response.json()
        assert "Missing signature" in data["error"]

    @pytest.mark.asyncio
    async def test_error_when_auth_token_not_configured(
        self, client, test_payload, monkeypatch
    ):
        """Test webhook returns 500 if auth token not configured."""
        monkeypatch.setattr(settings, "twilio_validate_signature", True)
        monkeypatch.setattr(settings, "twilio_auth_token", None)

        response = client.post(
            "/webhooks/twilio",
            data=test_payload,
            headers={"X-Twilio-Signature": "some_signature"}
        )

        assert response.status_code == 500


class TestTwilioModels:
    """Tests for Twilio Pydantic models."""

    def test_phone_extraction(self):
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

    def test_profile_name_fallback(self):
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

        payload.ProfileName = "John Doe"
        assert payload.get_profile_name() == "John Doe"
