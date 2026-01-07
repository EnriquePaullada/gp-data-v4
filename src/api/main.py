"""
FastAPI Application
Main entry point for webhook API and admin endpoints.
"""
import uuid
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import Response, JSONResponse
from contextlib import asynccontextmanager
from loguru import logger

from src.core.conversation_orchestrator import ConversationOrchestrator, SecurityException
from src.api.models.twilio import TwilioWebhookPayload
from src.config import settings
from src.repositories import db_manager
from src.services.twilio_service import twilio_service
from src.message_queue import InMemoryQueue, QueueWorker, QueuedMessage
from src.utils.rate_limiter import InMemoryRateLimiter


# Message processing handler
async def process_webhook_message(message: QueuedMessage) -> None:
    """
    Process a queued webhook message.

    This function is called by the queue worker for each message.
    It handles the full 3-agent pipeline and sends the response.

    Args:
        message: Queued webhook message

    Raises:
        Exception: If processing fails (triggers retry logic)
    """
    from src.api.main import app

    orchestrator: ConversationOrchestrator = app.state.orchestrator

    # Load or create lead from database
    lead = await orchestrator.lead_repo.get_or_create(
        phone_number=message.phone,
        full_name=message.profile_name
    )

    # Process message through 3-agent pipeline
    result = await orchestrator.process_message(message.body, lead)

    logger.info(
        f"✅ Message processed successfully",
        extra={
            "phone": message.phone,
            "intent": result.classification.intent,
            "action": result.strategy.action,
            "duration_ms": result.total_duration_ms
        }
    )

    # Send response via Twilio Messages API
    await twilio_service.send_whatsapp_message(
        to_number=message.phone,
        message=result.outbound_message
    )


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle: startup and shutdown events.

    Startup:
    - Initialize ConversationOrchestrator
    - Connect to MongoDB
    - Create database indexes
    - Initialize message queue
    - Start background worker

    Shutdown:
    - Stop background worker
    - Disconnect from MongoDB
    - Cleanup resources
    """
    logger.info("🚀 Starting GP Data API server...")

    # Initialize orchestrator
    orchestrator = ConversationOrchestrator()
    await orchestrator.initialize()

    # Initialize message queue
    queue = InMemoryQueue()

    # Initialize rate limiter
    rate_limiter = InMemoryRateLimiter(
        max_requests=settings.rate_limit_max_requests,
        window_seconds=settings.rate_limit_window_seconds,
        spike_threshold=settings.rate_limit_spike_threshold,
        spike_window_seconds=settings.rate_limit_spike_window_seconds,
        ban_duration_seconds=settings.rate_limit_ban_duration_seconds
    )

    # Start background worker
    worker = QueueWorker(
        queue=queue,
        handler=process_webhook_message,
        max_concurrent=10,
        poll_interval=1.0
    )

    # Store in app state for access in routes
    app.state.orchestrator = orchestrator
    app.state.queue = queue
    app.state.worker = worker
    app.state.rate_limiter = rate_limiter

    # Start worker in background
    import asyncio
    worker_task = asyncio.create_task(worker.start())
    app.state.worker_task = worker_task

    logger.info("✅ API server ready to receive webhooks")

    yield

    # Shutdown
    logger.info("🔌 Shutting down API server...")

    # Stop worker
    await worker.stop()

    # Cancel worker task if still running
    if not worker_task.done():
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass

    await orchestrator.shutdown()
    logger.info("✅ Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="GP Data v4 API",
    description="WhatsApp sales assistant API with 3-agent architecture",
    version="1.2.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    """
    Basic health check endpoint.

    Returns 200 if service is running.
    Used by load balancers and monitoring systems.
    """
    return {
        "status": "healthy",
        "service": "gp-data-v4",
        "version": "1.2.0"
    }


@app.get("/ready")
async def readiness_check(request: Request):
    """
    Readiness probe - checks if service can handle requests.

    Verifies:
    - MongoDB connection is active
    - Orchestrator is initialized

    Returns 200 if ready, 503 if not ready.
    """
    try:
        # Check orchestrator is initialized
        orchestrator = request.app.state.orchestrator
        if not orchestrator.lead_repo or not orchestrator.message_repo:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "reason": "Repositories not initialized"
                }
            )

        # Check MongoDB connection
        await db_manager.client.admin.command("ping")

        return {
            "status": "ready",
            "mongodb": "connected",
            "orchestrator": "initialized"
        }

    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "reason": str(e)
            }
        )


@app.post("/webhooks/twilio")
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
    MessagingServiceSid: str = Form(None)
):
    """
    Twilio WhatsApp webhook endpoint.

    Flow:
    1. Receive incoming WhatsApp message from Twilio
    2. Enqueue message for async processing
    3. Return 200 OK immediately to acknowledge receipt
    4. Background worker processes through 3-agent pipeline
    5. Background worker sends response via Twilio

    Request format: application/x-www-form-urlencoded (Twilio standard)
    Response format: JSON (acknowledgment)

    Args:
        All parameters from Twilio webhook payload

    Returns:
        JSON response acknowledging receipt

    Note:
        This endpoint returns immediately after enqueueing.
        Actual processing happens asynchronously in the background.
    """
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
        f"📱 Incoming WhatsApp message",
        extra={
            "phone": phone,
            "message_sid": payload.MessageSid,
            "body_length": len(payload.Body)
        }
    )

    try:
        # Get rate limiter and queue from app state
        rate_limiter: InMemoryRateLimiter = request.app.state.rate_limiter
        queue: InMemoryQueue = request.app.state.queue

        # Check if lead is banned
        is_banned = await rate_limiter.is_banned(phone)
        if is_banned:
            ban_info = await rate_limiter.get_ban_info(phone)
            if ban_info:
                ban_until, ban_reason = ban_info

                logger.warning(
                    f"🚫 Blocked message from banned lead",
                    extra={"phone": phone, "ban_reason": ban_reason}
                )

                # Send ban notification to user
                await twilio_service.send_whatsapp_message(
                    to_number=phone,
                    message=f"Your account is temporarily restricted. Please try again later."
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
                f"⚠️ Rate limit exceeded",
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
                    f"🚫 Lead auto-banned due to spike",
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

        # Create queued message
        queued_message = QueuedMessage(
            id=str(uuid.uuid4()),
            phone=phone,
            body=payload.Body,
            profile_name=profile_name,
            message_sid=payload.MessageSid
        )

        # Enqueue for async processing
        message_id = await queue.enqueue(queued_message)

        logger.info(
            f"✅ Message enqueued for processing",
            extra={
                "phone": phone,
                "message_id": message_id,
                "queue_id": queued_message.id,
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
                "status": "queued",
                "message_id": message_id,
                "phone": phone
            }
        )

    except Exception as e:
        logger.error(
            f"❌ Failed to enqueue webhook",
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
                f"❌ Failed to send error message",
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


@app.get("/metrics/queue")
async def queue_metrics(request: Request):
    """
    Get message queue metrics.

    Returns statistics about queue performance:
    - Pending messages
    - Messages being processed
    - Completed messages
    - Failed messages
    - Dead letter queue size
    - Average processing time
    - Error rate

    Returns:
        Queue metrics as JSON
    """
    try:
        queue: InMemoryQueue = request.app.state.queue
        metrics = await queue.get_metrics()

        return {
            "status": "ok",
            "metrics": metrics.model_dump()
        }

    except Exception as e:
        logger.error(f"Failed to get queue metrics: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e)
            }
        )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "GP Data v4 API",
        "version": "1.2.0",
        "endpoints": {
            "health": "/health",
            "ready": "/ready",
            "twilio_webhook": "/webhooks/twilio (POST)",
            "queue_metrics": "/metrics/queue"
        }
    }
