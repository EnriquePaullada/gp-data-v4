"""
Tests for ConversationOrchestrator
Verifies the complete 3-agent pipeline integration and orchestration logic.
"""
import pytest
import os
from unittest.mock import AsyncMock, MagicMock
from src.core.conversation_orchestrator import (
    ConversationOrchestrator,
    OrchestrationResult,
    HandoffActiveException
)
from src.models.lead import Lead, SalesStage, HandoffStatus
from src.models.message import MessageRole
from src.models.classifier_response import Intent, ClassifierResponse, UrgencyLevel
from src.models.director_response import StrategicAction, DirectorResponse, MessageStrategy
from src.models.intelligence import Sentiment
from src.services.handoff_service import HandoffService


@pytest.fixture
def test_lead():
    """Create a fresh test lead."""
    return Lead(
        lead_id="+5215538899800",
        full_name="Test User",
        current_stage=SalesStage.NEW
    )


@pytest.fixture
def orchestrator():
    """Create orchestrator instance."""
    return ConversationOrchestrator()


@pytest.mark.asyncio
class TestConversationOrchestrator:
    """Test suite for ConversationOrchestrator."""

    async def test_orchestrator_initialization(self, orchestrator):
        """Verify orchestrator initializes with all 3 agents."""
        assert orchestrator.classifier is not None
        assert orchestrator.director is not None
        assert orchestrator.executor is not None

    async def test_process_simple_greeting(self, orchestrator, test_lead):
        """Verify orchestrator processes a simple greeting."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("No API Key found")

        message = "Hello, I'm interested in your product"

        result = await orchestrator.process_message(message, test_lead)

        # Verify result structure
        assert isinstance(result, OrchestrationResult)
        assert result.outbound_message is not None
        assert len(result.outbound_message) > 0

        # Verify intermediate outputs exist
        assert result.classification is not None
        assert result.strategy is not None
        assert result.execution is not None

        # Verify lead was updated
        assert result.lead_updated.message_count == 2  # Incoming + outgoing
        assert len(result.lead_updated.recent_history) == 2

        # Verify timing
        assert result.total_duration_ms > 0

    async def test_lead_history_updated(self, orchestrator, test_lead):
        """Verify lead conversation history is updated correctly."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("No API Key found")

        initial_count = test_lead.message_count

        message = "I need pricing information"
        result = await orchestrator.process_message(message, test_lead)

        # Should add 2 messages: incoming (LEAD) + outgoing (ASSISTANT)
        assert result.lead_updated.message_count == initial_count + 2

        # Verify message roles
        history = result.lead_updated.recent_history
        assert history[-2].role == MessageRole.LEAD
        assert history[-2].content == message
        assert history[-1].role == MessageRole.ASSISTANT
        assert history[-1].content == result.outbound_message

    async def test_signals_extracted_and_added(self, orchestrator, test_lead):
        """Verify signals from classification are added to lead."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("No API Key found")

        initial_signals = len(test_lead.signals)

        message = "We have a team of 30 people and need HubSpot integration"
        result = await orchestrator.process_message(message, test_lead)

        # Classifier should extract signals about team size
        final_signals = len(result.lead_updated.signals)

        # May or may not extract signals depending on classification
        # Just verify the mechanism works (signals can be added)
        assert final_signals >= initial_signals

    async def test_classification_output(self, orchestrator, test_lead):
        """Verify classification step produces valid output."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("No API Key found")

        message = "How much does it cost?"

        result = await orchestrator.process_message(message, test_lead)

        # Verify classification structure
        classification = result.classification
        assert classification.intent in Intent
        assert 0 <= classification.intent_confidence <= 1.0
        assert classification.language in ["spanish", "english", "mixed"]
        assert classification.reasoning is not None

    async def test_strategy_output(self, orchestrator, test_lead):
        """Verify strategic decision step produces valid output."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("No API Key found")

        message = "I want to schedule a demo"

        result = await orchestrator.process_message(message, test_lead)

        # Verify strategy structure
        strategy = result.strategy
        assert strategy.action in StrategicAction
        assert strategy.message_strategy is not None
        assert strategy.message_strategy.tone is not None
        assert len(strategy.message_strategy.key_points) > 0
        assert strategy.strategic_reasoning is not None

    async def test_execution_output(self, orchestrator, test_lead):
        """Verify message execution step produces valid output."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("No API Key found")

        message = "Tell me about your features"

        result = await orchestrator.process_message(message, test_lead)

        # Verify execution structure
        execution = result.execution
        assert execution.message is not None
        assert execution.message.content is not None
        assert len(execution.message.content) > 0
        assert 0 <= execution.agreement_level <= 1.0
        assert execution.execution_summary is not None

    async def test_multiple_message_sequence(self, orchestrator, test_lead):
        """Verify orchestrator handles multiple sequential messages."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("No API Key found")

        messages = [
            "Hello",
            "We have 25 employees",
            "What's the pricing?"
        ]

        for i, message in enumerate(messages, 1):
            result = await orchestrator.process_message(message, test_lead)

            # Each iteration should increase message count by 2
            assert result.lead_updated.message_count == i * 2

            # Verify lead reference is updated
            test_lead = result.lead_updated

    async def test_batch_processing(self, orchestrator):
        """Verify batch processing handles multiple leads."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("No API Key found")

        # Create multiple leads with messages
        batch = [
            ("Hello", Lead(lead_id="+1", full_name="User 1")),
            ("Pricing?", Lead(lead_id="+2", full_name="User 2")),
            ("Demo please", Lead(lead_id="+3", full_name="User 3"))
        ]

        results = await orchestrator.process_batch(batch)

        # Should process all 3
        assert len(results) == 3

        # Each should have valid output
        for result in results:
            assert result.outbound_message is not None
            assert result.lead_updated.message_count == 2

    async def test_conversation_context_awareness(self, orchestrator, test_lead):
        """Verify orchestrator uses conversation history for context."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("No API Key found")

        # First message: establish context
        result1 = await orchestrator.process_message(
            "We're a medical company with 40 employees",
            test_lead
        )
        test_lead = result1.lead_updated

        # Second message: reference previous context
        result2 = await orchestrator.process_message(
            "What features do you have for our industry?",
            test_lead
        )

        # The classifier should have access to previous context
        # (40 employees should be in conversation history)
        assert len(result2.lead_updated.recent_history) == 4  # 2 exchanges

    async def test_orchestrator_with_custom_agents(self, test_lead):
        """Verify orchestrator accepts custom agent instances."""
        from src.agents.classifier_agent import ClassifierAgent
        from src.agents.director_agent import DirectorService
        from src.agents.executor_agent import ExecutorService

        custom_classifier = ClassifierAgent()
        custom_director = DirectorService(playbook_version="test")
        custom_executor = ExecutorService()

        orchestrator = ConversationOrchestrator(
            classifier=custom_classifier,
            director=custom_director,
            executor=custom_executor
        )

        assert orchestrator.classifier is custom_classifier
        assert orchestrator.director is custom_director
        assert orchestrator.executor is custom_executor

    async def test_performance_tracking(self, orchestrator, test_lead):
        """Verify orchestrator tracks performance metrics."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("No API Key found")

        message = "Quick test"

        result = await orchestrator.process_message(message, test_lead)

        # Should track duration
        assert result.total_duration_ms > 0
        # Should be reasonably fast (< 30 seconds even with retries)
        assert result.total_duration_ms < 30_000

    async def test_empty_message_handling(self, orchestrator, test_lead):
        """Verify orchestrator handles edge cases gracefully."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("No API Key found")

        message = ""

        # Should still process (agents will handle empty messages)
        result = await orchestrator.process_message(message, test_lead)

        assert result is not None
        assert result.outbound_message is not None

    # ============================================
    # PERSISTENCE INTEGRATION TESTS
    # ============================================

    async def test_initialize_creates_repositories(self):
        """Verify initialize() creates repository instances."""
        orchestrator = ConversationOrchestrator()

        # Before initialization
        assert orchestrator.lead_repo is None
        assert orchestrator.message_repo is None

        # Initialize
        await orchestrator.initialize()

        # After initialization
        assert orchestrator.lead_repo is not None
        assert orchestrator.message_repo is not None

        # Cleanup
        await orchestrator.shutdown()

    async def test_shutdown_cleans_up_resources(self):
        """Verify shutdown() properly cleans up resources."""
        orchestrator = ConversationOrchestrator()

        await orchestrator.initialize()

        # Verify repositories exist
        assert orchestrator.lead_repo is not None
        assert orchestrator.message_repo is not None

        # Shutdown
        await orchestrator.shutdown()

        # Verify repositories are cleared
        assert orchestrator.lead_repo is None
        assert orchestrator.message_repo is None

    async def test_process_message_persists_lead(self):
        """Verify process_message persists lead to database."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("No API Key found")

        orchestrator = ConversationOrchestrator()
        await orchestrator.initialize()

        try:
            # Create and persist a lead
            lead = await orchestrator.lead_repo.get_or_create(
                phone_number="+1234567890",
                full_name="Test User"
            )

            initial_message_count = lead.message_count

            # Process a message
            message = "Hello, I need help with pricing"
            result = await orchestrator.process_message(message, lead)

            # Reload lead from database
            reloaded_lead = await orchestrator.lead_repo.get_by_phone("+1234567890")

            # Verify lead was persisted with updates
            assert reloaded_lead is not None
            assert reloaded_lead.message_count == initial_message_count + 2
            assert reloaded_lead.last_interaction_at is not None

        finally:
            await orchestrator.shutdown()

    async def test_process_message_persists_messages(self):
        """Verify process_message persists messages to database."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("No API Key found")

        orchestrator = ConversationOrchestrator()
        await orchestrator.initialize()

        try:
            # Create lead
            lead = await orchestrator.lead_repo.get_or_create(
                phone_number="+9876543210",
                full_name="Message Test User"
            )

            # Process a message
            message_content = "I want to schedule a demo"
            result = await orchestrator.process_message(message_content, lead)

            # Query messages from database
            messages = await orchestrator.message_repo.get_conversation_history(
                lead_id="+9876543210",
                limit=10
            )

            # Should have 2 messages: incoming (LEAD) + outgoing (ASSISTANT)
            assert len(messages) >= 2

            # Verify incoming message
            lead_message = next((m for m in messages if m.role == MessageRole.LEAD), None)
            assert lead_message is not None
            assert lead_message.content == message_content

            # Verify assistant message
            assistant_message = next((m for m in messages if m.role == MessageRole.ASSISTANT), None)
            assert assistant_message is not None
            assert assistant_message.content == result.outbound_message

        finally:
            await orchestrator.shutdown()

    async def test_persisted_lead_can_be_reloaded(self):
        """Verify full round-trip: persist lead, reload, verify state."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("No API Key found")

        orchestrator = ConversationOrchestrator()
        await orchestrator.initialize()

        try:
            phone = "+5551234567"

            # First conversation turn
            lead = await orchestrator.lead_repo.get_or_create(
                phone_number=phone,
                full_name="Roundtrip Test"
            )

            result1 = await orchestrator.process_message(
                "We have 30 employees",
                lead
            )

            # Reload lead from database
            reloaded_lead = await orchestrator.lead_repo.get_by_phone(phone)

            assert reloaded_lead is not None
            assert reloaded_lead.message_count == 2
            assert reloaded_lead.full_name == "Roundtrip Test"

            # Continue conversation with reloaded lead
            result2 = await orchestrator.process_message(
                "What's the pricing?",
                reloaded_lead
            )

            # Reload again
            final_lead = await orchestrator.lead_repo.get_by_phone(phone)

            # Should have 4 messages total (2 exchanges)
            assert final_lead.message_count == 4

            # Verify message history in database
            all_messages = await orchestrator.message_repo.get_conversation_history(
                lead_id=phone,
                limit=10
            )

            assert len(all_messages) == 4

        finally:
            await orchestrator.shutdown()

    async def test_process_message_without_initialize_still_works(self, test_lead):
        """Verify orchestrator works without initialize (for unit tests)."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("No API Key found")

        # Create orchestrator WITHOUT calling initialize()
        orchestrator = ConversationOrchestrator()

        # Should still process messages (just without persistence)
        result = await orchestrator.process_message(
            "Test message",
            test_lead
        )

        # Verify pipeline still works
        assert result is not None
        assert result.outbound_message is not None
        assert result.lead_updated.message_count == 2

        # Repositories should still be None
        assert orchestrator.lead_repo is None
        assert orchestrator.message_repo is None

    async def test_multiple_messages_accumulate_in_database(self):
        """Verify multiple messages properly accumulate in database."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("No API Key found")

        orchestrator = ConversationOrchestrator()
        await orchestrator.initialize()

        try:
            phone = "+1112223333"
            lead = await orchestrator.lead_repo.get_or_create(
                phone_number=phone,
                full_name="Multi Message Test"
            )

            messages = [
                "Hello",
                "We have 50 employees",
                "Need HubSpot integration"
            ]

            # Process multiple messages
            for msg in messages:
                result = await orchestrator.process_message(msg, lead)
                lead = result.lead_updated

            # Verify final state in database
            final_lead = await orchestrator.lead_repo.get_by_phone(phone)
            assert final_lead.message_count == 6  # 3 incoming + 3 assistant

            # Verify all messages in database
            all_messages = await orchestrator.message_repo.get_conversation_history(
                lead_id=phone,
                limit=20
            )

            assert len(all_messages) == 6

            # Verify alternating roles
            roles = [m.role for m in all_messages]
            assert roles[0] == MessageRole.LEAD
            assert roles[1] == MessageRole.ASSISTANT
            assert roles[2] == MessageRole.LEAD
            assert roles[3] == MessageRole.ASSISTANT

        finally:
            await orchestrator.shutdown()

    async def test_signals_persisted_with_lead(self):
        """Verify BANT signals are persisted with lead state."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("No API Key found")

        orchestrator = ConversationOrchestrator()
        await orchestrator.initialize()

        try:
            phone = "+7778889999"
            lead = await orchestrator.lead_repo.get_or_create(
                phone_number=phone,
                full_name="Signal Test"
            )

            # Message that should generate signals
            result = await orchestrator.process_message(
                "We're a team of 40 people and need pricing for HubSpot integration",
                lead
            )

            # Reload from database
            reloaded_lead = await orchestrator.lead_repo.get_by_phone(phone)

            # Verify signals were persisted (if any were extracted)
            # Note: Signal extraction depends on classifier, might be 0 or more
            assert reloaded_lead.signals is not None
            assert isinstance(reloaded_lead.signals, list)

        finally:
            await orchestrator.shutdown()


# ============================================
# HUMAN HANDOFF TESTS
# ============================================

@pytest.mark.asyncio
class TestOrchestratorHandoff:
    """Test suite for human handoff integration in orchestrator."""

    async def test_handoff_active_raises_exception(self):
        """Verify orchestrator raises exception when lead is in handoff state."""
        lead = Lead(lead_id="+1234567890", full_name="Handoff Test")
        lead.request_handoff("Customer requested human")

        orchestrator = ConversationOrchestrator()

        with pytest.raises(HandoffActiveException) as exc_info:
            await orchestrator.process_message("Hello", lead)

        assert "human handoff state" in str(exc_info.value)

    async def test_handoff_triggered_on_escalate_action(self):
        """Verify handoff is triggered when Director returns ESCALATE."""
        lead = Lead(lead_id="+1234567890", full_name="Escalate Test")

        # Mock classifier
        mock_classifier = MagicMock()
        mock_classifier.classify = AsyncMock(return_value=ClassifierResponse(
            intent=Intent.SUPPORT,
            intent_confidence=0.9,
            topic="angry customer",
            topic_confidence=0.9,
            urgency=UrgencyLevel.HIGH,
            urgency_confidence=0.9,
            language="english",
            sentiment=Sentiment.NEGATIVE,
            engagement_level="high",
            requires_human_escalation=True,
            reasoning="Customer is upset and needs human attention",
            new_signals=[]
        ))

        # Mock director to return ESCALATE
        mock_director = MagicMock()
        mock_director.decide_next_move = AsyncMock(return_value=DirectorResponse(
            action=StrategicAction.ESCALATE,
            strategic_reasoning="Customer is frustrated, needs human touch",
            message_strategy=MessageStrategy(
                tone="empathetic",
                language="english",
                empathy_points=["Acknowledge frustration"],
                key_points=["Transfer to human"],
                conversational_goal="Smooth handoff"
            )
        ))

        # Mock handoff service
        mock_handoff = MagicMock(spec=HandoffService)
        mock_handoff.initiate_handoff = AsyncMock(return_value=True)
        mock_handoff.get_handoff_message = MagicMock(
            return_value="I'm connecting you with a specialist..."
        )

        orchestrator = ConversationOrchestrator(
            classifier=mock_classifier,
            director=mock_director,
            executor=None,  # Should not be called
            handoff_service=mock_handoff
        )

        result = await orchestrator.process_message("I'm very frustrated!", lead)

        # Verify handoff was triggered
        assert result.handoff_triggered is True
        assert result.execution is None  # Executor not called
        mock_handoff.initiate_handoff.assert_called_once()

    async def test_handoff_updates_lead_status(self):
        """Verify lead handoff status is updated when handoff triggered."""
        lead = Lead(lead_id="+1234567890", full_name="Status Test")

        # Mock classifier
        mock_classifier = MagicMock()
        mock_classifier.classify = AsyncMock(return_value=ClassifierResponse(
            intent=Intent.SUPPORT,
            intent_confidence=0.9,
            topic="escalation request",
            topic_confidence=0.9,
            urgency=UrgencyLevel.HIGH,
            urgency_confidence=0.9,
            language="english",
            sentiment=Sentiment.NEGATIVE,
            engagement_level="high",
            requires_human_escalation=True,
            reasoning="Needs human attention",
            new_signals=[]
        ))

        # Mock director to return ESCALATE
        mock_director = MagicMock()
        mock_director.decide_next_move = AsyncMock(return_value=DirectorResponse(
            action=StrategicAction.ESCALATE,
            strategic_reasoning="Escalating to human",
            message_strategy=MessageStrategy(
                tone="empathetic",
                language="english",
                empathy_points=["Understanding"],
                key_points=["Transfer"],
                conversational_goal="Handoff"
            )
        ))

        # Use real handoff service (not mocked) to test state update
        orchestrator = ConversationOrchestrator(
            classifier=mock_classifier,
            director=mock_director,
            executor=None
        )

        result = await orchestrator.process_message("Please transfer me", lead)

        # Verify lead status was updated
        assert result.lead_updated.handoff_status == HandoffStatus.REQUESTED
        assert result.lead_updated.handoff_reason is not None

    async def test_handoff_returns_appropriate_message(self):
        """Verify handoff returns the handoff message to the lead."""
        lead = Lead(lead_id="+1234567890", full_name="Message Test")

        mock_classifier = MagicMock()
        mock_classifier.classify = AsyncMock(return_value=ClassifierResponse(
            intent=Intent.SUPPORT,
            intent_confidence=0.9,
            topic="test",
            topic_confidence=0.9,
            urgency=UrgencyLevel.MEDIUM,
            urgency_confidence=0.9,
            language="spanish",
            sentiment=Sentiment.NEUTRAL,
            engagement_level="medium",
            requires_human_escalation=True,
            reasoning="Test",
            new_signals=[]
        ))

        mock_director = MagicMock()
        mock_director.decide_next_move = AsyncMock(return_value=DirectorResponse(
            action=StrategicAction.ESCALATE,
            strategic_reasoning="Test escalation",
            message_strategy=MessageStrategy(
                tone="empathetic",
                language="spanish",  # Spanish language
                empathy_points=["Test"],
                key_points=["Test"],
                conversational_goal="Handoff"
            )
        ))

        orchestrator = ConversationOrchestrator(
            classifier=mock_classifier,
            director=mock_director,
            executor=None
        )

        result = await orchestrator.process_message("Necesito ayuda", lead)

        # Verify Spanish handoff message is returned
        assert "especialista" in result.outbound_message.lower()
        assert result.handoff_triggered is True

    async def test_handoff_adds_message_to_history(self):
        """Verify handoff message is added to lead's conversation history."""
        lead = Lead(lead_id="+1234567890", full_name="History Test")
        initial_message_count = lead.message_count

        mock_classifier = MagicMock()
        mock_classifier.classify = AsyncMock(return_value=ClassifierResponse(
            intent=Intent.SUPPORT,
            intent_confidence=0.9,
            topic="test",
            topic_confidence=0.9,
            urgency=UrgencyLevel.MEDIUM,
            urgency_confidence=0.9,
            language="english",
            sentiment=Sentiment.NEUTRAL,
            engagement_level="medium",
            requires_human_escalation=True,
            reasoning="Test",
            new_signals=[]
        ))

        mock_director = MagicMock()
        mock_director.decide_next_move = AsyncMock(return_value=DirectorResponse(
            action=StrategicAction.ESCALATE,
            strategic_reasoning="Test",
            message_strategy=MessageStrategy(
                tone="empathetic",
                language="english",
                empathy_points=["Test"],
                key_points=["Test"],
                conversational_goal="Handoff"
            )
        ))

        orchestrator = ConversationOrchestrator(
            classifier=mock_classifier,
            director=mock_director,
            executor=None
        )

        result = await orchestrator.process_message("Help me", lead)

        # Verify messages were added (incoming + handoff response)
        assert result.lead_updated.message_count == initial_message_count + 2
        assert result.lead_updated.recent_history[-1].role == MessageRole.ASSISTANT
        assert result.lead_updated.recent_history[-2].role == MessageRole.LEAD

    async def test_normal_flow_not_affected_by_handoff(self):
        """Verify normal flow works when handoff is not triggered."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("No API Key found")

        lead = Lead(lead_id="+1234567890", full_name="Normal Test")
        orchestrator = ConversationOrchestrator()

        result = await orchestrator.process_message("Hello, tell me about your product", lead)

        # Verify normal flow completed
        assert result.handoff_triggered is False
        assert result.execution is not None
        assert result.outbound_message is not None
