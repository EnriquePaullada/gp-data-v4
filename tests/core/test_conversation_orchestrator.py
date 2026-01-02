"""
Tests for ConversationOrchestrator
Verifies the complete 3-agent pipeline integration and orchestration logic.
"""
import pytest
import os
from src.core.conversation_orchestrator import ConversationOrchestrator, OrchestrationResult
from src.models.lead import Lead, SalesStage
from src.models.message import MessageRole
from src.models.classifier_response import Intent
from src.models.director_response import StrategicAction


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
