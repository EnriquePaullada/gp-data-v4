"""
Twilio Webhook Signature Verification

Implements HMAC-SHA256 signature verification for Twilio webhooks
to prevent unauthorized requests and replay attacks.
"""

import hmac
import hashlib
import base64
from typing import Dict
from loguru import logger


class TwilioSignatureValidator:
    """
    Validates Twilio webhook signatures using HMAC-SHA256.

    Twilio signs all webhook requests with your auth token to ensure
    they are authentic. This validator verifies those signatures.

    Reference: https://www.twilio.com/docs/usage/security#validating-requests
    """

    def __init__(self, auth_token: str):
        """
        Initialize signature validator.

        Args:
            auth_token: Twilio auth token from account settings
        """
        self.auth_token = auth_token

    def validate(
        self,
        url: str,
        params: Dict[str, str],
        signature: str
    ) -> bool:
        """
        Validate Twilio webhook signature.

        Args:
            url: Full URL of the webhook endpoint (including query params)
            params: Form parameters from the webhook request
            signature: X-Twilio-Signature header value

        Returns:
            True if signature is valid, False otherwise
        """
        try:
            # Compute expected signature
            expected_signature = self.compute_signature(url, params)

            # Constant-time comparison to prevent timing attacks
            return hmac.compare_digest(expected_signature, signature)

        except Exception as e:
            logger.error(f"Signature validation error: {e}", exc_info=True)
            return False

    def compute_signature(self, url: str, params: Dict[str, str]) -> str:
        """
        Compute HMAC-SHA256 signature for Twilio webhook.

        Twilio's signature algorithm:
        1. Take the full URL (including query parameters)
        2. Append each POST parameter name and value (sorted by name)
        3. Sign the resulting string with HMAC-SHA256 using auth token
        4. Base64 encode the result

        Args:
            url: Full webhook URL
            params: Form parameters (dict)

        Returns:
            Base64-encoded HMAC-SHA256 signature
        """
        # Start with the full URL
        data = url

        # Append parameters in sorted order
        for key in sorted(params.keys()):
            data += key + params[key]

        # Compute HMAC-SHA256
        mac = hmac.new(
            self.auth_token.encode('utf-8'),
            data.encode('utf-8'),
            hashlib.sha256
        )

        # Return base64-encoded signature
        return base64.b64encode(mac.digest()).decode('utf-8')


def validate_twilio_signature(
    url: str,
    params: Dict[str, str],
    signature: str,
    auth_token: str
) -> bool:
    """
    Convenience function to validate Twilio signature.

    Args:
        url: Full webhook URL
        params: Form parameters
        signature: X-Twilio-Signature header
        auth_token: Twilio auth token

    Returns:
        True if valid, False otherwise
    """
    validator = TwilioSignatureValidator(auth_token)
    return validator.validate(url, params, signature)
