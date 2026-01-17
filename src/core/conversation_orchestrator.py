"""
Conversation Orchestrator
The central hub that coordinates the 3-agent pipeline for processing conversations.

Architecture:
    Incoming Message → Classifier → Director → Executor → Outbound Message
"""
from dataclasses import dataclass
from loguru import logger
from src.models.lead import Lead
from src.models.message import Message, MessageRole
from src.models.classifier_response import ClassifierResponse
from src.models.director_response import DirectorResponse
from src.models.executor_response import ExecutorResponse
from src.agents.classifier_agent import ClassifierAgent
from src.agents.director_agent import DirectorService
from src.agents.executor_agent import ExecutorService
from src.models.director_response import StrategicAction
from src.utils.observability import log_agent_execution
from src.repositories import db_manager, LeadRepository, MessageRepository
from src.utils.security_validator import SecurityValidator, ValidationResult
from src.services.handoff_service import HandoffService, get_handoff_service
from src.config import settings
from typing import Optional
import time


class SecurityException(Exception):
    """Raised when a security threat is detected."""
    pass


class HandoffActiveException(Exception):
    """Raised when lead is in active handoff state."""
    pass


@dataclass
class OrchestrationResult:
    """
    The complete result of processing a conversation turn.
    Contains all intermediate outputs for observability and debugging.
    """
    # Final output
    outbound_message: str

    # Intermediate agent outputs
    classification: ClassifierResponse
    strategy: DirectorResponse
    execution: Optional[ExecutorResponse]  # None when handoff triggered

    # Metadata
    total_duration_ms: float
    lead_updated: Lead

    # Handoff indicator
    handoff_triggered: bool = False


class ConversationOrchestrator:
    """
    Orchestrates the complete conversation processing pipeline.

    Responsibilities:
    1. Coordinate the 3-agent flow (Classifier → Director → Executor)
    2. Update lead state with new messages and signals
    3. Handle errors gracefully with fallbacks
    4. Provide observability into the orchestration process

    Usage:
        >>> orchestrator = ConversationOrchestrator()
        >>> result = await orchestrator.process_message(
        ...     message_content="I need pricing for 20 users",
        ...     lead=lead
        ... )
        >>> print(result.outbound_message)
    """

    def __init__(
        self,
        classifier: ClassifierAgent | None = None,
        director: DirectorService | None = None,
        executor: ExecutorService | None = None,
        handoff_service: HandoffService | None = None
    ):
        """
        Initialize the orchestrator with agent instances.

        Args:
            classifier: ClassifierAgent instance (creates new if None)
            director: DirectorService instance (creates new if None)
            executor: ExecutorService instance (creates new if None)
            handoff_service: HandoffService instance (creates new if None)
        """
        # Allow dependency injection for testing
        self.classifier = classifier or ClassifierAgent()
        self.director = director or DirectorService()
        self.executor = executor or ExecutorService()
        self.handoff_service = handoff_service or get_handoff_service()

        # Initialize repository instances
        self.lead_repo: Optional[LeadRepository] = None
        self.message_repo: Optional[MessageRepository] = None

        # Security validator
        self.security_validator = SecurityValidator(
            max_message_length=settings.max_message_length
        ) if settings.enable_security_validation else None

        logger.info("Conversation Orchestrator initialized with 3-agent pipeline")

    async def initialize(self) -> None:
        """
        Initialize MongoDB connection and repository instances.
        
        Must be called before processing messages in production.
        For testing, this is optional - orchestrator will work without persistence.

        Usage:
            >>> orchestrator = ConversationOrchestrator()
            >>> await orchestrator.initialize()
            >>> # Now ready to process messages with persistence
        """
        logger.info("Initializing ConversationOrchestrator with MongoDB persistence")

        # Connect to MongoDB
        await db_manager.connect()

        # Create indexes for optimal query performance
        await db_manager.create_indexes()

        # Initialize repository instances
        db = db_manager.database
        self.lead_repo = LeadRepository(db)
        self.message_repo = MessageRepository(db)

        logger.info("✅ ConversationOrchestrator initialized with persistence layer")

    async def shutdown(self) -> None:
        """
        Gracefully shut down MongoDB connection.
        Should be called when application is terminating.

        Usage:
            >>> await orchestrator.shutdown()
        """
        logger.info("Shutting down ConversationOrchestrator")
        await db_manager.disconnect()
        self.lead_repo = None
        self.message_repo = None
        logger.info("✅ ConversationOrchestrator shutdown complete")

    async def process_message(
        self,
        message_content: str,
        lead: Lead
    ) -> OrchestrationResult:
        """
        Process an incoming message through the complete 3-agent pipeline.

        This is the main entry point for conversation processing. It:
        0. Validates message security (prompt injection, PII, flooding)
        1. Adds the message to lead history
        2. Classifies the message (Agent 1)
        3. Determines strategy (Agent 2)
        4. Generates response (Agent 3)
        5. Updates lead state
        6. Persists to MongoDB (if enabled)

        Args:
            message_content: The incoming message text
            lead: The lead object (with conversation history)

        Returns:
            OrchestrationResult with complete pipeline outputs

        Raises:
            SecurityException: If security threats detected and message blocked
            Exception: If any critical step fails and fallbacks don't work
        """
        start_time = time.time()

        logger.info(f"🎬 Starting orchestration for lead: {lead.lead_id}")

        # Check if lead is in active handoff state
        if lead.is_handed_off:
            logger.info(
                f"Lead {lead.lead_id} is in handoff state, skipping AI processing",
                extra={"handoff_status": lead.handoff_status}
            )
            raise HandoffActiveException(
                f"Lead {lead.lead_id} is currently in human handoff state"
            )

        # Step 0: Security validation (if enabled)
        validation_result: Optional[ValidationResult] = None
        if self.security_validator:
            validation_result = self.security_validator.validate_message(
                message_content,
                lead.lead_id
            )

            if validation_result.should_block:
                logger.warning(
                    f"🚫 Message blocked due to security threats: {lead.lead_id}",
                    extra={
                        "lead_id": lead.lead_id,
                        "threats": [t.threat_type for t in validation_result.threats],
                        "threat_count": len(validation_result.threats)
                    }
                )
                # Raise exception to block processing
                raise SecurityException(
                    f"Message blocked due to security threats: "
                    f"{', '.join(str(t.threat_type) for t in validation_result.threats)}"
                )

            # Use sanitized message if PII was redacted
            if validation_result.sanitized_message != message_content:
                logger.info(
                    f"Message sanitized (PII redacted): {lead.lead_id}",
                    extra={"lead_id": lead.lead_id}
                )
                message_content = validation_result.sanitized_message

        # Step 1: Add incoming message to lead history
        incoming_message = Message(
            lead_id=lead.lead_id,
            role=MessageRole.LEAD,
            content=message_content
        )
        lead.add_message(incoming_message)

        try:
            # Step 2: CLASSIFY the message
            logger.info("Step 2/4: Classifying message")
            classification = await self.classifier.classify(message_content, lead)

            # Update lead with new signals
            for signal in classification.new_signals:
                lead.add_signal(signal)

            # Step 3: DIRECT - determine strategy
            logger.info("Step 3/4: Determining strategy")
            strategy = await self.director.decide_next_move(lead, classification)

            # Step 3.5: Check for ESCALATE action - trigger handoff
            if strategy.action == StrategicAction.ESCALATE:
                logger.info(f"Director triggered ESCALATE for lead {lead.lead_id}")

                # Initiate handoff with Director's reasoning
                await self.handoff_service.initiate_handoff(
                    lead=lead,
                    reason=strategy.strategic_reasoning,
                    urgency="high" if classification.urgency == "high" else "normal"
                )

                # Get handoff message in appropriate language
                language = strategy.message_strategy.language
                handoff_message = self.handoff_service.get_handoff_message(language)

                # Add handoff message to lead history
                assistant_message = Message(
                    lead_id=lead.lead_id,
                    role=MessageRole.ASSISTANT,
                    content=handoff_message
                )
                lead.add_message(assistant_message)

                # Persist if repositories available
                if self.lead_repo and self.message_repo:
                    await self.lead_repo.save(lead)
                    await self.message_repo.save_messages([incoming_message, assistant_message])

                total_duration_ms = (time.time() - start_time) * 1000

                log_agent_execution(
                    agent_name="ConversationOrchestrator",
                    lead_id=lead.lead_id,
                    action="handoff_triggered",
                    duration_ms=total_duration_ms,
                    intent=classification.intent,
                    strategic_action=strategy.action
                )

                logger.success(
                    f"🤝 Handoff triggered in {total_duration_ms:.0f}ms | "
                    f"Reason: {strategy.strategic_reasoning[:50]}..."
                )

                # Return early with handoff result (no Executor needed)
                return OrchestrationResult(
                    outbound_message=handoff_message,
                    classification=classification,
                    strategy=strategy,
                    execution=None,  # No executor for handoff
                    total_duration_ms=total_duration_ms,
                    lead_updated=lead,
                    handoff_triggered=True
                )

            # Step 4: EXECUTE - generate response
            logger.info("Step 4/4: Generating response")
            execution = await self.executor.craft_message(lead, strategy)

            # Step 5: Add assistant response to lead history
            assistant_message = Message(
                lead_id=lead.lead_id,
                role=MessageRole.ASSISTANT,
                content=execution.message.content
            )
            lead.add_message(assistant_message)

            # Step 6: Persist to MongoDB (if repositories initialized)
            if self.lead_repo and self.message_repo:
                logger.debug("Persisting lead and messages to MongoDB")

                # Persist updated lead state
                await self.lead_repo.save(lead)

                # Persist both incoming and assistant messages
                await self.message_repo.save_messages([incoming_message, assistant_message])

                logger.debug(
                    f"✅ Persisted lead and 2 messages to MongoDB",
                    extra={
                        "lead_id": lead.lead_id,
                        "message_count": lead.message_count,
                        "signals_count": len(lead.signals)
                    }
                )

            # Calculate total duration
            total_duration_ms = (time.time() - start_time) * 1000

            # Log orchestration completion
            log_agent_execution(
                agent_name="ConversationOrchestrator",
                lead_id=lead.lead_id,
                action="process_message",
                duration_ms=total_duration_ms,
                intent=classification.intent,
                strategic_action=strategy.action,
                agreement_level=execution.agreement_level
            )

            logger.success(
                f"✅ Orchestration complete in {total_duration_ms:.0f}ms | "
                f"Intent: {classification.intent} | Action: {strategy.action}"
            )

            return OrchestrationResult(
                outbound_message=execution.message.content,
                classification=classification,
                strategy=strategy,
                execution=execution,
                total_duration_ms=total_duration_ms,
                lead_updated=lead
            )

        except Exception as e:
            total_duration_ms = (time.time() - start_time) * 1000
            logger.error(f"❌ Orchestration failed after {total_duration_ms:.0f}ms: {e}")
            raise

    async def process_batch(
        self,
        messages: list[tuple[str, Lead]]
    ) -> list[OrchestrationResult]:
        """
        Process multiple messages in sequence.

        Useful for:
        - Bulk processing queued messages
        - Testing multiple scenarios
        - Batch operations

        Args:
            messages: List of (message_content, lead) tuples

        Returns:
            List of OrchestrationResults
        """
        results = []

        for message_content, lead in messages:
            try:
                result = await self.process_message(message_content, lead)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process message for {lead.lead_id}: {e}")
                # Continue processing other messages
                continue

        return results
