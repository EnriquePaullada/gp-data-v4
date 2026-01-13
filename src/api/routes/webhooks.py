"""
Webhook Endpoints

Twilio WhatsApp webhook handler with signature validation,
rate limiting, and message buffering for burst handling.
"""
from fastapi import APIRouter, Request, Form
from fastapi.responses import JSONResponse
from loguru import logger

from src.api.models.twilio import TwilioWebhookPayload
from src.config import settings
from src.services.twilio_service import twilio_service
from src.message_queue.buffer import MessageBuffer
from src.utils.rate_limiter import InMemoryRateLimiter
from src.utils.twilio_signature import validate_twilio_signature

router = APIRouter(prefix="/webhooks", tags=["Webhooks"])


@router.post("/twilio")
async def twilio_webhook(
    request: Request,
    MessageSid: str = Form(...),
    AccountSid: str = Form(...),
    From: str = Form(...),
    To: str = Form(...),
    Body: str = Form(...),
    NumMedia: str = Form(default="0"),
    ProfileName: str = Form(None),
    WaId: str = Form(None),
    ApiVersion: str = Form(None),
    SmsStatus: str = Form(None),
    MessagingServiceSid: str = Form(None),
    X_Twilio_Signature: str = Form(None)
):
    """
    Twilio WhatsApp webhook endpoint.

    Flow:
    1. Validate Twilio webhook signature (security)
    2. Receive incoming WhatsApp message from Twilio
    3. Enqueue message for async processing
    4. Return 200 OK immediately to acknowledge receipt
    5. Background worker processes through 3-agent pipeline
    6. Background worker sends response via Twilio

    Request format: application/x-www-form-urlencoded (Twilio standard)
    Response format: JSON (acknowledgment)

    Args:
        All parameters from Twilio webhook payload
        X_Twilio_Signature: Signature header for validation

    Returns:
        JSON response acknowledging receipt

    Note:
        This endpoint returns immediately after enqueueing.
        Actual processing happens asynchronously in the background.
    """
    # Validate Twilio signature if enabled
    if settings.twilio_validate_signature:
        if not settings.twilio_auth_token:
            logger.error("TWILIO_AUTH_TOKEN not configured but signature validation is enabled")
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "error": "Webhook authentication not configured"
                }
            )

        # Get signature from header
        signature = request.headers.get("X-Twilio-Signature")

        if not signature:
            logger.warning("Missing X-Twilio-Signature header")
            return JSONResponse(
                status_code=401,
                content={
                    "status": "unauthorized",
                    "error": "Missing signature header"
                }
            )

        # Build params dict from form data
        params = {
            "MessageSid": MessageSid,
            "AccountSid": AccountSid,
            "From": From,
            "To": To,
            "Body": Body,
            "NumMedia": NumMedia,
        }
        if ProfileName:
            params["ProfileName"] = ProfileName
        if WaId:
            params["WaId"] = WaId
        if ApiVersion:
            params["ApiVersion"] = ApiVersion
        if SmsStatus:
            params["SmsStatus"] = SmsStatus
        if MessagingServiceSid:
            params["MessagingServiceSid"] = MessagingServiceSid

        # Validate signature
        url = str(request.url)
        is_valid = validate_twilio_signature(
            url=url,
            params=params,
            signature=signature,
            auth_token=settings.twilio_auth_token
        )

        if not is_valid:
            logger.warning(
                "Invalid Twilio signature",
                extra={"url": url}
            )
            return JSONResponse(
                status_code=401,
                content={
                    "status": "unauthorized",
                    "error": "Invalid signature"
                }
            )

        logger.debug("Twilio signature validated successfully")

    # Parse webhook payload
    payload = TwilioWebhookPayload(
        MessageSid=MessageSid,
        AccountSid=AccountSid,
        From=From,
        To=To,
        Body=Body,
        NumMedia=NumMedia,
        ProfileName=ProfileName,
        WaId=WaId,
        ApiVersion=ApiVersion,
        SmsStatus=SmsStatus,
        MessagingServiceSid=MessagingServiceSid
    )

    phone = payload.get_clean_phone()
    profile_name = payload.get_profile_name()

    logger.info(
        f"Incoming WhatsApp message",
        extra={
            "phone": phone,
            "message_sid": payload.MessageSid,
            "body_length": len(payload.Body)
        }
    )

    try:
        # Get rate limiter and message buffer from app state
        rate_limiter: InMemoryRateLimiter = request.app.state.rate_limiter
        message_buffer: MessageBuffer = request.app.state.message_buffer

        # Check if lead is banned
        is_banned = await rate_limiter.is_banned(phone)
        if is_banned:
            ban_info = await rate_limiter.get_ban_info(phone)
            if ban_info:
                ban_until, ban_reason = ban_info

                logger.warning(
                    f"Blocked message from banned lead",
                    extra={"phone": phone, "ban_reason": ban_reason}
                )

                # Send ban notification to user
                await twilio_service.send_whatsapp_message(
                    to_number=phone,
                    message="Your account is temporarily restricted. Please try again later."
                )

                return JSONResponse(
                    status_code=429,
                    content={
                        "status": "banned",
                        "phone": phone,
                        "ban_until": ban_until.isoformat(),
                        "reason": ban_reason
                    }
                )

        # Check rate limit
        rate_limit_result = await rate_limiter.check_rate_limit(phone)

        if not rate_limit_result.allowed:
            logger.warning(
                f"Rate limit exceeded",
                extra={
                    "phone": phone,
                    "reason": rate_limit_result.reason
                }
            )

            return JSONResponse(
                status_code=429,
                headers={
                    "X-RateLimit-Limit": str(settings.rate_limit_max_requests),
                    "X-RateLimit-Remaining": str(rate_limit_result.remaining),
                    "X-RateLimit-Reset": str(int(rate_limit_result.reset_at.timestamp())),
                    "Retry-After": str(rate_limit_result.retry_after)
                },
                content={
                    "status": "rate_limited",
                    "phone": phone,
                    "retry_after": rate_limit_result.retry_after,
                    "reason": rate_limit_result.reason
                }
            )

        # Detect spike and auto-ban if configured
        if settings.rate_limit_auto_ban_on_spike:
            spike_detected = await rate_limiter.detect_spike(phone)
            if spike_detected:
                await rate_limiter.ban_lead(
                    phone,
                    settings.rate_limit_ban_duration_seconds,
                    "Spike detected: Too many messages in short period"
                )

                logger.warning(
                    f"Lead auto-banned due to spike",
                    extra={"phone": phone}
                )

                await twilio_service.send_whatsapp_message(
                    to_number=phone,
                    message="You've been sending too many messages. Your account is temporarily restricted."
                )

                return JSONResponse(
                    status_code=429,
                    content={
                        "status": "banned",
                        "phone": phone,
                        "reason": "Spike detected"
                    }
                )

        # Add to message buffer (handles WhatsApp burst messages)
        # Buffer will concatenate rapid messages and enqueue after delay
        await message_buffer.add(
            phone=phone,
            body=payload.Body,
            message_sid=payload.MessageSid,
            profile_name=profile_name
        )

        logger.info(
            "Message buffered for processing",
            extra={
                "phone": phone,
                "message_sid": payload.MessageSid,
                "rate_limit_remaining": rate_limit_result.remaining
            }
        )

        return JSONResponse(
            status_code=200,
            headers={
                "X-RateLimit-Limit": str(settings.rate_limit_max_requests),
                "X-RateLimit-Remaining": str(rate_limit_result.remaining),
                "X-RateLimit-Reset": str(int(rate_limit_result.reset_at.timestamp()))
            },
            content={
                "status": "buffered",
                "message_sid": payload.MessageSid,
                "phone": phone
            }
        )

    except Exception as e:
        logger.error(
            f"Failed to enqueue webhook",
            extra={"phone": phone, "error": str(e)},
            exc_info=True
        )

        # Try to send immediate error message
        try:
            await twilio_service.send_whatsapp_message(
                to_number=phone,
                message="I'm experiencing technical difficulties. Please try again in a moment."
            )
        except Exception as send_error:
            logger.error(
                f"Failed to send error message",
                extra={"phone": phone, "error": str(send_error)},
                exc_info=True
            )

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "phone": phone,
                "error": str(e)
            }
        )
