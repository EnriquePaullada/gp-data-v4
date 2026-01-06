"""
Twilio WhatsApp messaging service.
Handles sending messages via Twilio Messages API.
"""
from twilio.rest import Client
from loguru import logger
from typing import Optional

from src.config import settings


class TwilioService:
    """
    Service for sending WhatsApp messages via Twilio API.

    This service is responsible for:
    - Sending outbound messages to leads
    - Handling Twilio API errors
    - Rate limiting and retries
    """

    def __init__(self):
        """Initialize Twilio client with credentials from settings."""
        # Allow for testing without credentials
        if settings.twilio_account_sid and settings.twilio_auth_token:
            self.client = Client(
                settings.twilio_account_sid,
                settings.twilio_auth_token
            )
        else:
            self.client = None
            logger.warning("Twilio credentials not configured - service will not be functional")

        self.from_number = settings.twilio_whatsapp_from

    async def send_whatsapp_message(
        self,
        to_number: str,
        message: str,
        max_retries: int = 3
    ) -> Optional[str]:
        """
        Send WhatsApp message via Twilio API.

        Args:
            to_number: Recipient's phone number (E.164 format, e.g., "+5215538899800")
            message: Message content to send
            max_retries: Maximum number of retry attempts

        Returns:
            Message SID if successful, None if failed

        Raises:
            Exception: If all retries fail or if Twilio client not configured
        """
        if not self.client:
            raise Exception("Twilio client not configured - missing credentials")

        # Ensure to_number has whatsapp: prefix
        if not to_number.startswith("whatsapp:"):
            to_number = f"whatsapp:{to_number}"

        logger.info(
            f"📤 Sending WhatsApp message",
            extra={
                "to": to_number,
                "message_length": len(message),
                "from": self.from_number
            }
        )

        for attempt in range(max_retries):
            try:
                # Send message via Twilio API
                message_response = self.client.messages.create(
                    body=message,
                    from_=self.from_number,
                    to=to_number
                )

                logger.info(
                    f"✅ Message sent successfully",
                    extra={
                        "message_sid": message_response.sid,
                        "to": to_number,
                        "status": message_response.status
                    }
                )

                return message_response.sid

            except Exception as e:
                logger.error(
                    f"❌ Failed to send message (attempt {attempt + 1}/{max_retries})",
                    extra={
                        "to": to_number,
                        "error": str(e),
                        "attempt": attempt + 1
                    },
                    exc_info=True
                )

                # If last attempt, raise the exception
                if attempt == max_retries - 1:
                    raise

        return None


# Global service instance
twilio_service = TwilioService()
