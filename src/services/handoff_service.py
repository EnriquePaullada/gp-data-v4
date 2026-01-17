"""
Human Handoff Service

Manages escalation from AI to human agents when the AI cannot
adequately handle a conversation. Implements a pluggable notification
system with Slack as the initial implementation.
"""

import httpx
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Protocol
from src.config import get_settings
from src.models.lead import Lead, HandoffStatus
from src.utils.observability import logger


@dataclass
class HandoffRequest:
    """Details of a handoff request."""
    lead_id: str
    lead_name: Optional[str]
    reason: str
    conversation_summary: str
    urgency: str = "normal"  # normal, high, critical


class HandoffNotifier(Protocol):
    """
    Protocol for handoff notification channels.

    Implement this to add new notification channels (email, SMS, etc.)
    """

    async def notify(self, request: HandoffRequest) -> bool:
        """
        Send handoff notification.

        Args:
            request: Handoff request details

        Returns:
            True if notification sent successfully
        """
        ...


class SlackHandoffNotifier:
    """
    Slack webhook implementation for handoff notifications.

    Sends formatted messages to a Slack channel when handoff is triggered.
    """

    def __init__(self, webhook_url: Optional[str] = None):
        settings = get_settings()
        self._webhook_url = webhook_url or settings.slack_handoff_webhook_url

    @property
    def is_configured(self) -> bool:
        """Check if Slack webhook is configured."""
        return self._webhook_url is not None

    async def notify(self, request: HandoffRequest) -> bool:
        """
        Send handoff notification to Slack.

        Args:
            request: Handoff request details

        Returns:
            True if notification sent successfully
        """
        if not self._webhook_url:
            logger.warning("Slack webhook not configured, skipping notification")
            return False

        # Build Slack message payload
        payload = self._build_slack_payload(request)

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self._webhook_url,
                    json=payload,
                    timeout=10.0
                )
                response.raise_for_status()

            logger.info(
                f"Slack handoff notification sent for lead {request.lead_id}",
                extra={"lead_id": request.lead_id, "reason": request.reason}
            )
            return True

        except httpx.HTTPError as e:
            logger.error(
                f"Failed to send Slack notification: {e}",
                extra={"lead_id": request.lead_id, "error": str(e)}
            )
            return False

    def _build_slack_payload(self, request: HandoffRequest) -> dict:
        """Build Slack Block Kit message payload."""
        urgency_emoji = {
            "normal": ":hand:",
            "high": ":warning:",
            "critical": ":rotating_light:"
        }.get(request.urgency, ":hand:")

        return {
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"{urgency_emoji} Human Handoff Requested",
                        "emoji": True
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Lead:*\n{request.lead_name or 'Unknown'}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Phone:*\n{request.lead_id}"
                        }
                    ]
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Reason:*\n{request.reason}"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Recent Conversation:*\n```{request.conversation_summary[:500]}```"
                    }
                },
                {
                    "type": "divider"
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": "Reply in WhatsApp to take over this conversation"
                        }
                    ]
                }
            ]
        }


class LogOnlyNotifier:
    """
    Fallback notifier that only logs handoff requests.

    Used when no external notification channel is configured.
    """

    async def notify(self, request: HandoffRequest) -> bool:
        """Log handoff request."""
        logger.warning(
            f"Handoff requested (no notifier configured): {request.lead_id}",
            extra={
                "lead_id": request.lead_id,
                "lead_name": request.lead_name,
                "reason": request.reason,
                "urgency": request.urgency
            }
        )
        return True


class HandoffService:
    """
    Coordinates human handoff workflow.

    Responsibilities:
    1. Update lead handoff state
    2. Send notifications to human agents
    3. Provide handoff status and metrics

    Usage:
        service = HandoffService()

        # Trigger handoff
        success = await service.initiate_handoff(
            lead=lead,
            reason="Complex technical question"
        )

        # Assign to human
        service.assign_to_agent(lead, "agent_123")

        # Resolve handoff
        service.resolve(lead)
    """

    def __init__(self, notifier: Optional[HandoffNotifier] = None):
        """
        Initialize handoff service.

        Args:
            notifier: Notification channel (defaults to Slack if configured)
        """
        if notifier:
            self._notifier = notifier
        else:
            slack = SlackHandoffNotifier()
            self._notifier = slack if slack.is_configured else LogOnlyNotifier()

    async def initiate_handoff(
        self,
        lead: Lead,
        reason: str,
        urgency: str = "normal"
    ) -> bool:
        """
        Initiate human handoff for a lead.

        Updates lead state and sends notification to human agents.

        Args:
            lead: Lead requiring handoff
            reason: Why handoff is needed
            urgency: Priority level (normal, high, critical)

        Returns:
            True if handoff initiated and notification sent
        """
        # Update lead state
        lead.request_handoff(reason)

        logger.info(
            f"Initiating handoff for lead {lead.lead_id}",
            extra={"lead_id": lead.lead_id, "reason": reason, "urgency": urgency}
        )

        # Build handoff request
        request = HandoffRequest(
            lead_id=lead.lead_id,
            lead_name=lead.full_name,
            reason=reason,
            conversation_summary=lead.format_history(limit=5),
            urgency=urgency
        )

        # Send notification
        return await self._notifier.notify(request)

    def assign_to_agent(self, lead: Lead, agent_id: str) -> None:
        """
        Assign handoff to a human agent.

        Args:
            lead: Lead being handed off
            agent_id: Identifier for human agent
        """
        lead.assign_handoff(agent_id)
        logger.info(
            f"Handoff assigned to {agent_id}",
            extra={"lead_id": lead.lead_id, "agent_id": agent_id}
        )

    def resolve(self, lead: Lead) -> None:
        """
        Mark handoff as resolved.

        Args:
            lead: Lead whose handoff is complete
        """
        lead.resolve_handoff()
        logger.info(
            f"Handoff resolved for lead {lead.lead_id}",
            extra={"lead_id": lead.lead_id}
        )

    def cancel(self, lead: Lead) -> None:
        """
        Cancel handoff and return to AI handling.

        Args:
            lead: Lead to return to AI
        """
        lead.clear_handoff()
        logger.info(
            f"Handoff cancelled for lead {lead.lead_id}",
            extra={"lead_id": lead.lead_id}
        )

    def get_handoff_message(self, language: str = "english") -> str:
        """
        Get the message to send to lead when handoff is triggered.

        Args:
            language: Message language (english or spanish)

        Returns:
            Human handoff message for the lead
        """
        messages = {
            "english": (
                "I'm connecting you with one of our specialists who can better "
                "assist you. They'll be with you shortly!"
            ),
            "spanish": (
                "Te estoy conectando con uno de nuestros especialistas que "
                "podrá ayudarte mejor. ¡Estarán contigo en breve!"
            )
        }
        return messages.get(language, messages["english"])


# Singleton instance
_service: Optional[HandoffService] = None


def get_handoff_service() -> HandoffService:
    """Get or create the handoff service singleton."""
    global _service
    if _service is None:
        _service = HandoffService()
    return _service
