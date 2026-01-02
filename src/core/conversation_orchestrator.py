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
from src.utils.observability import log_agent_execution
import time


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
    execution: ExecutorResponse

    # Metadata
    total_duration_ms: float
    lead_updated: Lead


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
        executor: ExecutorService | None = None
    ):
        """
        Initialize the orchestrator with agent instances.

        Args:
            classifier: ClassifierAgent instance (creates new if None)
            director: DirectorService instance (creates new if None)
            executor: ExecutorService instance (creates new if None)
        """
        # Allow dependency injection for testing
        self.classifier = classifier or ClassifierAgent()
        self.director = director or DirectorService()
        self.executor = executor or ExecutorService()

        logger.info("ConversationOrchestrator initialized with 3-agent pipeline")

    async def process_message(
        self,
        message_content: str,
        lead: Lead
    ) -> OrchestrationResult:
        """
        Process an incoming message through the complete 3-agent pipeline.

        This is the main entry point for conversation processing. It:
        1. Adds the message to lead history
        2. Classifies the message (Agent 1)
        3. Determines strategy (Agent 2)
        4. Generates response (Agent 3)
        5. Updates lead state
        6. Returns complete result

        Args:
            message_content: The incoming message text
            lead: The lead object (with conversation history)

        Returns:
            OrchestrationResult with complete pipeline outputs

        Raises:
            Exception: If any critical step fails and fallbacks don't work
        """
        start_time = time.time()

        logger.info(f"🎬 Starting orchestration for lead: {lead.lead_id}")

        # Step 0: Add incoming message to lead history
        incoming_message = Message(
            lead_id=lead.lead_id,
            role=MessageRole.LEAD,
            content=message_content
        )
        lead.add_message(incoming_message)

        try:
            # Step 1: CLASSIFY the message
            logger.info("Step 1/3: Classifying message")
            classification = await self.classifier.classify(message_content, lead)

            # Update lead with new signals
            for signal in classification.new_signals:
                lead.add_signal(signal)

            # Step 2: DIRECT - determine strategy
            logger.info("Step 2/3: Determining strategy")
            strategy = await self.director.decide_next_move(lead, classification)

            # Step 3: EXECUTE - generate response
            logger.info("Step 3/3: Generating response")
            execution = await self.executor.craft_message(lead, strategy)

            # Step 4: Add assistant response to lead history
            assistant_message = Message(
                lead_id=lead.lead_id,
                role=MessageRole.ASSISTANT,
                content=execution.message.content
            )
            lead.add_message(assistant_message)

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
