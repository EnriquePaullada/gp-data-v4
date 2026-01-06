"""
FastAPI Application
Main entry point for webhook API and admin endpoints.
"""
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import Response, JSONResponse
from contextlib import asynccontextmanager
from loguru import logger

from src.core.conversation_orchestrator import ConversationOrchestrator, SecurityException
from src.api.models.twilio import TwilioWebhookPayload
from src.config import settings
from src.repositories import db_manager
from src.services.twilio_service import twilio_service


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle: startup and shutdown events.

    Startup:
    - Initialize ConversationOrchestrator
    - Connect to MongoDB
    - Create database indexes

    Shutdown:
    - Disconnect from MongoDB
    - Cleanup resources
    """
    logger.info("🚀 Starting GP Data API server...")

    # Initialize orchestrator
    orchestrator = ConversationOrchestrator()
    await orchestrator.initialize()

    # Store in app state for access in routes
    app.state.orchestrator = orchestrator

    logger.info("✅ API server ready to receive webhooks")

    yield

    # Shutdown
    logger.info("🔌 Shutting down API server...")
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
    2. Process through 3-agent pipeline (Classifier → Director → Executor)
    3. Send response via Twilio Messages API
    4. Return 200 OK to acknowledge receipt

    Request format: application/x-www-form-urlencoded (Twilio standard)
    Response format: JSON (acknowledgment)

    Args:
        All parameters from Twilio webhook payload

    Returns:
        JSON response acknowledging receipt

    Raises:
        HTTPException: If processing fails or security threat detected
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
        # Get orchestrator from app state
        orchestrator: ConversationOrchestrator = request.app.state.orchestrator

        # Load or create lead from database
        lead = await orchestrator.lead_repo.get_or_create(
            phone_number=phone,
            full_name=profile_name
        )

        # Process message through 3-agent pipeline
        result = await orchestrator.process_message(payload.Body, lead)

        logger.info(
            f"✅ Message processed successfully",
            extra={
                "phone": phone,
                "intent": result.classification.intent,
                "action": result.strategy.action,
                "duration_ms": result.total_duration_ms
            }
        )

        # Send response via Twilio Messages API
        message_sid = await twilio_service.send_whatsapp_message(
            to_number=phone,
            message=result.outbound_message
        )

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message_sid": message_sid,
                "phone": phone
            }
        )

    except SecurityException as e:
        logger.warning(
            f"🚫 Security threat detected",
            extra={"phone": phone, "error": str(e)}
        )

        # Send generic error message to user
        await twilio_service.send_whatsapp_message(
            to_number=phone,
            message="I'm sorry, but I cannot process that message. Please rephrase your question."
        )

        return JSONResponse(
            status_code=200,
            content={
                "status": "security_error",
                "phone": phone
            }
        )

    except Exception as e:
        logger.error(
            f"❌ Failed to process webhook",
            extra={"phone": phone, "error": str(e)},
            exc_info=True
        )

        # Send fallback error message
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
            status_code=200,
            content={
                "status": "error",
                "phone": phone,
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
            "twilio_webhook": "/webhooks/twilio (POST)"
        }
    }
