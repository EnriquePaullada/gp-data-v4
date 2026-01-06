"""
Pydantic models for Twilio webhook payloads.
"""
from pydantic import BaseModel, Field
from typing import Optional


class TwilioWebhookPayload(BaseModel):
    """
    Twilio WhatsApp webhook payload structure.

    See: https://www.twilio.com/docs/sms/twiml#twilios-request-to-your-application
    """
    # Message identifiers
    MessageSid: str = Field(..., description="Unique identifier for the message")
    AccountSid: str = Field(..., description="Twilio account identifier")
    MessagingServiceSid: Optional[str] = Field(None, description="Messaging service SID if applicable")

    # Sender information (the lead)
    From: str = Field(..., description="Sender's phone number (E.164 format with whatsapp: prefix)")
    To: str = Field(..., description="Your Twilio WhatsApp number")

    # Message content
    Body: str = Field(..., description="Message text content")
    NumMedia: str = Field(default="0", description="Number of media items attached")

    # Optional fields
    ProfileName: Optional[str] = Field(None, description="WhatsApp profile name of sender")
    WaId: Optional[str] = Field(None, description="WhatsApp ID (phone number without whatsapp: prefix)")

    # Metadata
    ApiVersion: Optional[str] = Field(None, description="Twilio API version")
    SmsStatus: Optional[str] = Field(None, description="Message status")

    def get_clean_phone(self) -> str:
        """
        Extract clean E.164 phone number from WhatsApp format.

        Twilio sends: "whatsapp:+5215538899800"
        Returns: "+5215538899800"
        """
        return self.From.replace("whatsapp:", "")

    def get_profile_name(self) -> str:
        """Get sender's name, defaulting to phone if ProfileName not available."""
        return self.ProfileName or self.get_clean_phone()
