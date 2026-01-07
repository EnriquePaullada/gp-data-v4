"""
FastAPI Dependencies

Reusable dependencies for request validation and authentication.
"""

from fastapi import Request, HTTPException, status, Header
from typing import Optional
from loguru import logger

from src.config import settings
from src.utils.twilio_signature import validate_twilio_signature


async def validate_twilio_webhook_signature(
    request: Request,
    x_twilio_signature: Optional[str] = Header(None)
) -> None:
    """
    Dependency to validate Twilio webhook signature.

    Verifies that the request is authentically from Twilio by validating
    the X-Twilio-Signature header using HMAC-SHA256.

    Args:
        request: FastAPI request object
        x_twilio_signature: X-Twilio-Signature header value

    Raises:
        HTTPException: 401 if signature is invalid or missing

    Note:
        Can be disabled via TWILIO_VALIDATE_SIGNATURE=false for testing
    """
    # Skip validation if disabled (for testing)
    if not settings.twilio_validate_signature:
        logger.warning("⚠️ Twilio signature validation is DISABLED")
        return

    # Check if auth token is configured
    if not settings.twilio_auth_token:
        logger.error("❌ TWILIO_AUTH_TOKEN not configured but signature validation is enabled")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Webhook authentication not configured"
        )

    # Check if signature header is present
    if not x_twilio_signature:
        logger.warning("🚫 Missing X-Twilio-Signature header")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing signature header"
        )

    # Get full URL (including scheme, host, and path)
    url = str(request.url)

    # Get form parameters
    form_data = await request.form()
    params = {key: value for key, value in form_data.items()}

    # Validate signature
    is_valid = validate_twilio_signature(
        url=url,
        params=params,
        signature=x_twilio_signature,
        auth_token=settings.twilio_auth_token
    )

    if not is_valid:
        logger.warning(
            "🚫 Invalid Twilio signature",
            extra={
                "url": url,
                "signature": x_twilio_signature[:20] + "..."  # Truncate for logging
            }
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid signature"
        )

    logger.debug("✅ Twilio signature validated successfully")
