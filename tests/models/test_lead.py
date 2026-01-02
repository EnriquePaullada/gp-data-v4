import pytest
import datetime as dt
from src.models.lead import Lead
from src.models.message import Message, MessageRole
from src.models.intelligence import IntelligenceSignal, BANTDimension, ConfidenceScore

# --- FIXTURES ---

@pytest.fixture
def fresh_lead():
    return Lead(
        lead_id="+5215538899800",
        full_name="Enrique Paullada"
    )

@pytest.fixture
def message_factory():
    """Returns a function to create messages with incremental IDs."""
    def _create_message(lead_id: str, index: int):
        msg = Message(
            lead_id=lead_id,
            role=MessageRole.LEAD if index % 2 == 0 else MessageRole.ASSISTANT,
            content=f"Test message {index}",
            timestamp=dt.datetime.now(dt.UTC)
        )
        msg.id = f"msg_{index}"
        return msg
    return _create_message


# --- TESTS ---

def test_sliding_window_memory(fresh_lead, message_factory):
    """Verifies that the lead only keeps the most recent 20 messages."""
    for i in range(25):
        msg = message_factory(fresh_lead.lead_id, i)
        fresh_lead.add_message(msg)
    
    assert len(fresh_lead.recent_history) == 20
    assert fresh_lead.message_count == 25
    # The first message in history should be index 5 (since we added 25 and kept 20)
    assert fresh_lead.recent_history[0].content == "Test message 5"

def test_heartbeat_sync(fresh_lead, message_factory):
    """Ensures lead heartbeat perfectly matches the latest message timestamp."""
    msg = message_factory(fresh_lead.lead_id, 1)
    fresh_lead.add_message(msg)
    
    assert fresh_lead.last_interaction_at == msg.timestamp

def test_deterministic_bant_reduction(fresh_lead):
    """Verifies that the latest intelligence signal overwrites previous ones."""
    sig_low = IntelligenceSignal(
        dimension=BANTDimension.BUDGET,
        extracted_value="low",
        confidence=ConfidenceScore(value=1.0, reasoning="Direct"),
        source_message_id="msg_1",
        raw_evidence="Budget is low"
    )
    sig_high = IntelligenceSignal(
        dimension=BANTDimension.BUDGET,
        extracted_value="high",
        confidence=ConfidenceScore(value=0.8, reasoning="Pivot"),
        source_message_id="msg_2",
        raw_evidence="Budget is high now"
    )

    fresh_lead.add_signal(sig_low)
    fresh_lead.add_signal(sig_high)

    assert fresh_lead.bant_summary[BANTDimension.BUDGET] == "high"
    assert len(fresh_lead.signals) == 2

def test_inference_traceability(fresh_lead):
    """Verifies that signals can track multiple source messages."""
    msg_ids = ["msg_1", "msg_2", "msg_3"]
    sig = IntelligenceSignal(
        dimension=BANTDimension.AUTHORITY,
        extracted_value="high",
        confidence=ConfidenceScore(value=0.9, reasoning="Inferred"),
        source_message_id=msg_ids[-1],
        is_inferred=True,
        inferred_from=msg_ids,
        raw_evidence="I am the decision maker"
    )
    fresh_lead.add_signal(sig)

    last_signal = fresh_lead.signals[-1]
    assert last_signal.is_inferred is True
    assert last_signal.inferred_from == ["msg_1", "msg_2", "msg_3"]


def test_format_history_with_roles(fresh_lead):
    """Verifies format_history() includes role prefixes by default."""
    # Add some messages
    msg1 = Message(
        lead_id=fresh_lead.lead_id,
        role=MessageRole.LEAD,
        content="Hello, I need help"
    )
    msg2 = Message(
        lead_id=fresh_lead.lead_id,
        role=MessageRole.ASSISTANT,
        content="Sure, how can I assist?"
    )
    msg3 = Message(
        lead_id=fresh_lead.lead_id,
        role=MessageRole.LEAD,
        content="I want pricing info"
    )

    fresh_lead.add_message(msg1)
    fresh_lead.add_message(msg2)
    fresh_lead.add_message(msg3)

    formatted = fresh_lead.format_history()

    expected = (
        "LEAD: Hello, I need help\n"
        "ASSISTANT: Sure, how can I assist?\n"
        "LEAD: I want pricing info"
    )

    assert formatted == expected


def test_format_history_without_roles(fresh_lead):
    """Verifies format_history() can exclude role prefixes."""
    msg1 = Message(
        lead_id=fresh_lead.lead_id,
        role=MessageRole.LEAD,
        content="Hello"
    )
    msg2 = Message(
        lead_id=fresh_lead.lead_id,
        role=MessageRole.ASSISTANT,
        content="Hi there"
    )

    fresh_lead.add_message(msg1)
    fresh_lead.add_message(msg2)

    formatted = fresh_lead.format_history(include_roles=False)

    expected = "Hello\nHi there"
    assert formatted == expected


def test_format_history_with_limit(fresh_lead, message_factory):
    """Verifies format_history() respects message limit."""
    # Add 5 messages
    for i in range(5):
        msg = message_factory(fresh_lead.lead_id, i)
        fresh_lead.add_message(msg)

    # Request only last 3 messages
    formatted = fresh_lead.format_history(limit=3)

    # Should only contain last 3 messages (indices 2, 3, 4)
    assert "Test message 2" in formatted
    assert "Test message 3" in formatted
    assert "Test message 4" in formatted
    assert "Test message 0" not in formatted
    assert "Test message 1" not in formatted

    # Verify exactly 3 lines
    lines = formatted.strip().split('\n')
    assert len(lines) == 3


def test_format_history_empty(fresh_lead):
    """Verifies format_history() handles empty history."""
    formatted = fresh_lead.format_history()

    assert formatted == ""


def test_format_history_single_message(fresh_lead):
    """Verifies format_history() works with single message."""
    msg = Message(
        lead_id=fresh_lead.lead_id,
        role=MessageRole.LEAD,
        content="Single message"
    )
    fresh_lead.add_message(msg)

    formatted = fresh_lead.format_history()

    assert formatted == "LEAD: Single message"


def test_format_history_limit_exceeds_history(fresh_lead, message_factory):
    """Verifies format_history() handles limit larger than history."""
    # Add only 2 messages
    for i in range(2):
        msg = message_factory(fresh_lead.lead_id, i)
        fresh_lead.add_message(msg)

    # Request 10 messages (more than available)
    formatted = fresh_lead.format_history(limit=10)

    # Should return all available messages
    lines = formatted.strip().split('\n')
    assert len(lines) == 2