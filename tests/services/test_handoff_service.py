"""Tests for the Human Handoff Service."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

from src.models.lead import Lead, HandoffStatus
from src.services.handoff_service import (
    HandoffService,
    SlackHandoffNotifier,
    LogOnlyNotifier,
    HandoffRequest,
    get_handoff_service,
)


# --- FIXTURES ---

@pytest.fixture
def fresh_lead():
    """Returns a clean Lead for testing."""
    return Lead(lead_id="+5215538899800", full_name="Test User")


@pytest.fixture
def mock_notifier():
    """Returns a mock notifier."""
    notifier = AsyncMock()
    notifier.notify = AsyncMock(return_value=True)
    return notifier


@pytest.fixture
def handoff_service(mock_notifier):
    """Returns a HandoffService with mock notifier."""
    return HandoffService(notifier=mock_notifier)


# --- HANDOFF SERVICE TESTS ---

class TestHandoffService:
    """Tests for HandoffService class."""

    async def test_initiate_handoff_updates_lead_state(self, handoff_service, fresh_lead):
        """Verifies handoff initiation updates lead status."""
        await handoff_service.initiate_handoff(
            lead=fresh_lead,
            reason="Complex question"
        )

        assert fresh_lead.handoff_status == HandoffStatus.REQUESTED
        assert fresh_lead.handoff_reason == "Complex question"
        assert fresh_lead.handoff_requested_at is not None

    async def test_initiate_handoff_sends_notification(self, handoff_service, fresh_lead, mock_notifier):
        """Verifies notification is sent on handoff initiation."""
        await handoff_service.initiate_handoff(
            lead=fresh_lead,
            reason="Customer requested human"
        )

        mock_notifier.notify.assert_called_once()
        call_args = mock_notifier.notify.call_args[0][0]
        assert isinstance(call_args, HandoffRequest)
        assert call_args.lead_id == fresh_lead.lead_id
        assert call_args.reason == "Customer requested human"

    async def test_initiate_handoff_returns_notification_result(self, fresh_lead, mock_notifier):
        """Verifies initiate_handoff returns notification success status."""
        mock_notifier.notify = AsyncMock(return_value=True)
        service = HandoffService(notifier=mock_notifier)
        result = await service.initiate_handoff(fresh_lead, "Test")
        assert result is True

        mock_notifier.notify = AsyncMock(return_value=False)
        service = HandoffService(notifier=mock_notifier)
        result = await service.initiate_handoff(fresh_lead, "Test")
        assert result is False

    async def test_initiate_handoff_with_urgency(self, handoff_service, fresh_lead, mock_notifier):
        """Verifies urgency level is passed to notification."""
        await handoff_service.initiate_handoff(
            lead=fresh_lead,
            reason="Angry customer",
            urgency="critical"
        )

        call_args = mock_notifier.notify.call_args[0][0]
        assert call_args.urgency == "critical"

    def test_assign_to_agent(self, handoff_service, fresh_lead):
        """Verifies handoff can be assigned to an agent."""
        fresh_lead.request_handoff("Test")
        handoff_service.assign_to_agent(fresh_lead, "agent_123")

        assert fresh_lead.handoff_status == HandoffStatus.ASSIGNED
        assert fresh_lead.handoff_assigned_to == "agent_123"

    def test_resolve_handoff(self, handoff_service, fresh_lead):
        """Verifies handoff can be resolved."""
        fresh_lead.request_handoff("Test")
        fresh_lead.assign_handoff("agent_123")
        handoff_service.resolve(fresh_lead)

        assert fresh_lead.handoff_status == HandoffStatus.RESOLVED
        assert fresh_lead.is_handed_off is False

    def test_cancel_handoff(self, handoff_service, fresh_lead):
        """Verifies handoff can be cancelled."""
        fresh_lead.request_handoff("Test")
        handoff_service.cancel(fresh_lead)

        assert fresh_lead.handoff_status == HandoffStatus.NONE
        assert fresh_lead.handoff_reason is None

    def test_get_handoff_message_english(self, handoff_service):
        """Verifies English handoff message."""
        msg = handoff_service.get_handoff_message("english")
        assert "specialist" in msg.lower()
        assert "shortly" in msg.lower()

    def test_get_handoff_message_spanish(self, handoff_service):
        """Verifies Spanish handoff message."""
        msg = handoff_service.get_handoff_message("spanish")
        assert "especialista" in msg.lower()
        assert "breve" in msg.lower()

    def test_get_handoff_message_defaults_to_english(self, handoff_service):
        """Verifies unknown language defaults to English."""
        msg = handoff_service.get_handoff_message("french")
        assert "specialist" in msg.lower()


# --- SLACK NOTIFIER TESTS ---

class TestSlackHandoffNotifier:
    """Tests for SlackHandoffNotifier class."""

    def test_is_configured_with_url(self):
        """Verifies is_configured returns True when URL is set."""
        notifier = SlackHandoffNotifier(webhook_url="https://hooks.slack.com/test")
        assert notifier.is_configured is True

    def test_is_not_configured_without_url(self):
        """Verifies is_configured returns False when URL is not set."""
        with patch("src.services.handoff_service.get_settings") as mock_settings:
            mock_settings.return_value.slack_handoff_webhook_url = None
            notifier = SlackHandoffNotifier()
            assert notifier.is_configured is False

    async def test_notify_skips_when_not_configured(self):
        """Verifies notification is skipped when webhook not configured."""
        with patch("src.services.handoff_service.get_settings") as mock_settings:
            mock_settings.return_value.slack_handoff_webhook_url = None
            notifier = SlackHandoffNotifier()

            request = HandoffRequest(
                lead_id="+1234567890",
                lead_name="Test",
                reason="Test reason",
                conversation_summary="Hello"
            )
            result = await notifier.notify(request)
            assert result is False

    async def test_notify_sends_http_request(self):
        """Verifies HTTP request is sent to Slack webhook."""
        notifier = SlackHandoffNotifier(webhook_url="https://hooks.slack.com/test")

        request = HandoffRequest(
            lead_id="+1234567890",
            lead_name="Test User",
            reason="Complex question",
            conversation_summary="Recent messages here"
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=MagicMock(status_code=200))
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            result = await notifier.notify(request)

            assert result is True
            mock_instance.post.assert_called_once()
            call_args = mock_instance.post.call_args
            assert call_args[0][0] == "https://hooks.slack.com/test"

    async def test_notify_handles_http_error(self):
        """Verifies HTTP errors are handled gracefully."""
        notifier = SlackHandoffNotifier(webhook_url="https://hooks.slack.com/test")

        request = HandoffRequest(
            lead_id="+1234567890",
            lead_name="Test",
            reason="Test",
            conversation_summary="Test"
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(side_effect=httpx.HTTPError("Connection failed"))
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            result = await notifier.notify(request)
            assert result is False

    def test_build_slack_payload_structure(self):
        """Verifies Slack payload has correct Block Kit structure."""
        notifier = SlackHandoffNotifier(webhook_url="https://hooks.slack.com/test")

        request = HandoffRequest(
            lead_id="+1234567890",
            lead_name="John Doe",
            reason="Pricing negotiation",
            conversation_summary="Hello, I need help with pricing",
            urgency="high"
        )

        payload = notifier._build_slack_payload(request)

        assert "blocks" in payload
        blocks = payload["blocks"]
        assert len(blocks) >= 4

        # Header block
        assert blocks[0]["type"] == "header"
        assert ":warning:" in blocks[0]["text"]["text"]  # High urgency emoji

        # Fields should include lead name and phone
        fields_block = next(b for b in blocks if b["type"] == "section" and "fields" in b)
        field_texts = [f["text"] for f in fields_block["fields"]]
        assert any("John Doe" in t for t in field_texts)
        assert any("+1234567890" in t for t in field_texts)


# --- LOG ONLY NOTIFIER TESTS ---

class TestLogOnlyNotifier:
    """Tests for LogOnlyNotifier class."""

    async def test_notify_returns_true(self):
        """Verifies log notifier always returns True."""
        notifier = LogOnlyNotifier()
        request = HandoffRequest(
            lead_id="+1234567890",
            lead_name="Test",
            reason="Test",
            conversation_summary="Test"
        )
        result = await notifier.notify(request)
        assert result is True


# --- SINGLETON TESTS ---

class TestHandoffServiceSingleton:
    """Tests for singleton behavior."""

    def test_get_handoff_service_returns_same_instance(self):
        """Verifies singleton returns same instance."""
        # Reset singleton for test
        import src.services.handoff_service as module
        module._service = None

        service1 = get_handoff_service()
        service2 = get_handoff_service()

        assert service1 is service2

        # Reset for other tests
        module._service = None
