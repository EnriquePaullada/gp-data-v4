"""
FastAPI Application

Main entry point for the GP Data v4 API.
Handles application lifecycle and router mounting.
"""
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from loguru import logger

from src.core.conversation_orchestrator import ConversationOrchestrator
from src.config import settings
from src.message_queue import InMemoryQueue, QueueWorker, QueuedMessage
from src.utils.rate_limiter import InMemoryRateLimiter
from src.services.twilio_service import twilio_service
from src.api.routes import health_router, webhooks_router, metrics_router


async def process_webhook_message(message: QueuedMessage) -> None:
    """
    Process a queued webhook message through the 3-agent pipeline.

    This function is called by the queue worker for each message.
    It handles lead loading, pipeline processing, and response sending.

    Args:
        message: Queued webhook message with phone, body, and metadata

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
        "Message processed successfully",
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle: startup and shutdown events.

    Startup:
    - Initialize ConversationOrchestrator (connects to MongoDB)
    - Initialize message queue and rate limiter
    - Start background queue worker

    Shutdown:
    - Stop background worker gracefully
    - Disconnect from MongoDB
    """
    logger.info("Starting GP Data API server...")

    # Initialize orchestrator (connects to MongoDB, creates indexes)
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

    # Create background worker
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
    worker_task = asyncio.create_task(worker.start())
    app.state.worker_task = worker_task

    logger.info("API server ready to receive webhooks")

    yield

    # Shutdown
    logger.info("Shutting down API server...")

    await worker.stop()

    if not worker_task.done():
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass

    await orchestrator.shutdown()
    logger.info("Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="GP Data v4 API",
    description="WhatsApp sales assistant API with 3-agent architecture",
    version="1.3.0",
    lifespan=lifespan
)

# Mount routers
app.include_router(health_router)
app.include_router(webhooks_router)
app.include_router(metrics_router)
