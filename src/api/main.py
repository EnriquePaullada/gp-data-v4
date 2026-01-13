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
from src.message_queue.buffer import MessageBuffer
from src.utils.rate_limiter import InMemoryRateLimiter
from src.services.twilio_service import twilio_service
from src.services.followup_scheduler import get_followup_scheduler, FollowUpType
from src.models.intelligence import IntelligenceSignal, BANTDimension, ConfidenceScore
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


async def on_buffer_flush(
    phone: str,
    combined_body: str,
    first_message_sid: str,
    profile_name: str | None
) -> None:
    """
    Callback when message buffer flushes for a lead.

    Creates a QueuedMessage from the buffered messages and enqueues
    it for processing through the 3-agent pipeline.

    Args:
        phone: Lead phone number (E.164)
        combined_body: Concatenated message bodies
        first_message_sid: SID of first message in burst
        profile_name: WhatsApp profile name
    """
    import uuid
    from src.api.main import app

    queue: InMemoryQueue = app.state.queue

    queued_message = QueuedMessage(
        id=str(uuid.uuid4()),
        phone=phone,
        body=combined_body,
        profile_name=profile_name,
        message_sid=first_message_sid
    )

    message_id = await queue.enqueue(queued_message)

    logger.info(
        "Buffered messages enqueued",
        extra={
            "phone": phone,
            "message_id": message_id,
            "body_length": len(combined_body)
        }
    )


async def run_followup_worker(poll_interval: float = 60.0) -> None:
    """
    Background worker that checks for and sends follow-up messages.

    Polls the database periodically for leads needing follow-up,
    generates appropriate messages, and sends via Twilio.

    Args:
        poll_interval: Seconds between polls (default 60)
    """
    from src.api.main import app

    scheduler = get_followup_scheduler()
    logger.info("Follow-up worker started", extra={"poll_interval": poll_interval})

    while True:
        try:
            await asyncio.sleep(poll_interval)

            orchestrator: ConversationOrchestrator = app.state.orchestrator
            leads = await orchestrator.lead_repo.get_leads_needing_followup(limit=10)

            if not leads:
                continue

            logger.info(f"Processing {len(leads)} leads for follow-up")

            for lead in leads:
                try:
                    action = scheduler.get_next_followup(lead)

                    if not action:
                        # No follow-up needed, clear the scheduled time
                        lead.next_followup_at = None
                        await orchestrator.lead_repo.save(lead)
                        continue

                    # Check if should mark cold instead
                    if scheduler.should_mark_cold(lead):
                        scheduler.mark_cold(lead)
                        await orchestrator.lead_repo.save(lead)
                        logger.info(f"Marked lead {lead.lead_id} as cold")
                        continue

                    # Get guidance for message generation
                    guidance = scheduler.get_followup_prompt_guidance(action.followup_type)

                    # Generate follow-up message using executor agent
                    followup_message = await _generate_followup_message(
                        orchestrator, lead, action.followup_type, guidance
                    )

                    # Send via Twilio
                    await twilio_service.send_whatsapp_message(
                        to_number=lead.lead_id,
                        message=followup_message
                    )

                    # Record follow-up signal
                    signal = IntelligenceSignal(
                        dimension=BANTDimension.NEED,
                        extracted_value="followup_sent",
                        confidence=ConfidenceScore(
                            value=1.0,
                            reasoning=f"followup_sent: {action.followup_type.value} attempt {action.attempt_number}"
                        ),
                        source_message_id=f"followup_{action.attempt_number}",
                        raw_evidence=f"[System: {action.followup_type.value} follow-up sent]"
                    )
                    lead.add_signal(signal)

                    # Schedule next follow-up
                    scheduler.schedule_followup(lead)
                    await orchestrator.lead_repo.save(lead)

                    logger.info(
                        f"Sent {action.followup_type.value} follow-up to {lead.lead_id}",
                        extra={"attempt": action.attempt_number}
                    )

                except Exception as e:
                    logger.error(f"Follow-up failed for {lead.lead_id}: {e}")
                    continue

        except asyncio.CancelledError:
            logger.info("Follow-up worker cancelled")
            break
        except Exception as e:
            logger.error(f"Follow-up worker error: {e}")
            await asyncio.sleep(poll_interval)


async def _generate_followup_message(
    orchestrator: ConversationOrchestrator,
    lead,
    followup_type: FollowUpType,
    guidance: dict
) -> str:
    """
    Generate a follow-up message for a lead.

    Uses the executor agent with follow-up context.
    """
    # Build context prompt for follow-up
    prompt = (
        f"Generate a {followup_type.value.replace('_', ' ')} follow-up message.\n"
        f"Tone: {guidance.get('tone', 'warm and professional')}\n"
        f"Goal: {guidance.get('conversational_goal', 'Re-engage the lead')}\n"
        f"Key points: {', '.join(guidance.get('key_points', []))}\n"
        f"Lead name: {lead.full_name or 'there'}\n"
        f"Lead stage: {lead.current_stage.value}"
    )

    try:
        # Use executor agent to generate message
        from src.agents.executor_agent import run_executor
        from src.models.director_response import DirectorResponse, StrategicAction, MessageStrategy

        # Create minimal director response for follow-up
        director_response = DirectorResponse(
            action=StrategicAction.ENGAGE,
            strategic_reasoning=f"Proactive {followup_type.value} follow-up",
            message_strategy=MessageStrategy(
                tone=guidance.get("tone", "warm"),
                empathy_points=guidance.get("empathy_points", []),
                key_points=guidance.get("key_points", []),
                questions_to_ask=[],
                conversational_goal=guidance.get("conversational_goal", "")
            )
        )

        result = await run_executor(
            director_response=director_response,
            lead=lead,
            conversation_history=[]
        )
        return result.message.content

    except Exception as e:
        logger.warning(f"Failed to generate follow-up via agent: {e}, using template")
        # Fallback to simple template
        name = lead.full_name.split()[0] if lead.full_name else "there"
        templates = {
            FollowUpType.CHECK_IN: f"Hola {name}, solo queria ver como estas. Estoy aqui si tienes alguna pregunta.",
            FollowUpType.VALUE_REMINDER: f"Hola {name}, recuerda que podemos ayudarte a optimizar tu proceso de ventas.",
            FollowUpType.URGENCY: f"Hola {name}, queria recordarte que tenemos disponibilidad limitada esta semana.",
            FollowUpType.SOCIAL_PROOF: f"Hola {name}, varios clientes como tu ya estan viendo resultados increibles.",
            FollowUpType.FINAL_ATTEMPT: f"Hola {name}, este es mi ultimo mensaje. La puerta siempre esta abierta.",
        }
        return templates.get(followup_type, templates[FollowUpType.CHECK_IN])


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

    # Initialize message buffer for WhatsApp burst handling
    message_buffer = MessageBuffer(on_flush=on_buffer_flush)

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
    app.state.message_buffer = message_buffer

    # Start workers in background
    worker_task = asyncio.create_task(worker.start())
    followup_task = asyncio.create_task(run_followup_worker(poll_interval=60.0))

    app.state.worker_task = worker_task
    app.state.followup_task = followup_task

    logger.info("API server ready to receive webhooks")

    yield

    # Shutdown
    logger.info("Shutting down API server...")

    # Flush any pending buffered messages before stopping worker
    logger.info("Flushing message buffer...")
    await message_buffer.flush_all()

    await worker.stop()

    # Stop background tasks
    for task, name in [(worker_task, "queue worker"), (followup_task, "followup worker")]:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.info(f"Stopped {name}")

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
