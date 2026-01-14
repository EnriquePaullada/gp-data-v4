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


# --- CONTEXT PRUNING TESTS ---

def test_format_history_no_pruning_when_under_limit(fresh_lead):
    """Verifies no pruning occurs when history is under max_context_chars."""
    # Add a few short messages (well under 6000 char limit)
    for i in range(5):
        msg = Message(
            lead_id=fresh_lead.lead_id,
            role=MessageRole.LEAD if i % 2 == 0 else MessageRole.ASSISTANT,
            content=f"Short message {i}"
        )
        fresh_lead.add_message(msg)

    formatted = fresh_lead.format_history()

    # Should contain all messages without pruning
    assert "Short message 0" in formatted
    assert "Short message 4" in formatted
    assert "[Earlier conversation:" not in formatted
    assert "truncated" not in formatted


def test_format_history_prunes_when_exceeds_limit(fresh_lead):
    """Verifies pruning occurs when history exceeds max_context_chars."""
    # Create messages with long content to exceed 6000 char limit
    long_content = "x" * 800  # 800 chars per message
    for i in range(10):  # 10 messages × 800 chars = 8000 chars
        msg = Message(
            lead_id=fresh_lead.lead_id,
            role=MessageRole.LEAD if i % 2 == 0 else MessageRole.ASSISTANT,
            content=f"Message {i}: {long_content}"
        )
        fresh_lead.add_message(msg)

    formatted = fresh_lead.format_history()

    # Should be significantly reduced from original 8000+ chars
    # Allow small buffer for summary lines and formatting (6100 chars)
    assert len(formatted) <= 6100

    # Original unpruned would be much longer
    assert len(formatted) < 8000

    # Should contain truncation indicator
    assert "[Earlier conversation:" in formatted or "messages truncated" in formatted or "messages omitted" in formatted


def test_format_history_preserves_recent_messages(fresh_lead):
    """Verifies recent messages are always kept in full when pruning."""
    # Create long messages to trigger pruning
    long_content = "x" * 800
    for i in range(10):
        msg = Message(
            lead_id=fresh_lead.lead_id,
            role=MessageRole.LEAD if i % 2 == 0 else MessageRole.ASSISTANT,
            content=f"Message {i}: {long_content}"
        )
        fresh_lead.add_message(msg)

    formatted = fresh_lead.format_history()

    # Recent messages (last 5 by default from config) should be complete
    # Message 9 (most recent) should be in full
    assert f"Message 9: {long_content}" in formatted
    # Message 8 should also be complete
    assert f"Message 8: {long_content}" in formatted


def test_format_history_truncates_older_messages(fresh_lead):
    """Verifies older messages are truncated when pruning."""
    # Create long messages
    long_content = "y" * 800
    for i in range(10):
        msg = Message(
            lead_id=fresh_lead.lead_id,
            role=MessageRole.LEAD if i % 2 == 0 else MessageRole.ASSISTANT,
            content=f"Message {i}: {long_content}"
        )
        fresh_lead.add_message(msg)

    formatted = fresh_lead.format_history()

    # Older messages should be either truncated with "..." or omitted
    # Message 0 (oldest) might be truncated
    if "Message 0:" in formatted:
        # If present, should be truncated (not full 800 chars)
        message_0_start = formatted.find("Message 0:")
        message_0_section = formatted[message_0_start:message_0_start + 850]
        # Should not contain the full long_content
        assert long_content not in message_0_section or "..." in message_0_section


def test_format_history_adds_pruning_summary(fresh_lead):
    """Verifies pruning summary is added when messages are omitted."""
    # Create very long messages to force aggressive pruning
    very_long_content = "z" * 1200
    for i in range(15):
        msg = Message(
            lead_id=fresh_lead.lead_id,
            role=MessageRole.LEAD if i % 2 == 0 else MessageRole.ASSISTANT,
            content=f"Message {i}: {very_long_content}"
        )
        fresh_lead.add_message(msg)

    formatted = fresh_lead.format_history()

    # Should include summary about pruning
    has_summary = (
        "[Earlier conversation:" in formatted or
        "messages truncated" in formatted or
        "messages omitted" in formatted or
        "context limit" in formatted
    )
    assert has_summary


def test_format_history_respects_min_recent_messages_config(fresh_lead):
    """Verifies min_recent_messages config is respected during pruning."""
    from src.config import get_settings
    settings = get_settings()

    # Create messages that will trigger pruning
    long_content = "a" * 900
    num_messages = 12
    for i in range(num_messages):
        msg = Message(
            lead_id=fresh_lead.lead_id,
            role=MessageRole.LEAD if i % 2 == 0 else MessageRole.ASSISTANT,
            content=f"Message {i}: {long_content}"
        )
        fresh_lead.add_message(msg)

    formatted = fresh_lead.format_history()

    # Last min_recent_messages (default 5) should be complete
    min_recent = min(settings.min_recent_messages, num_messages)
    for i in range(num_messages - min_recent, num_messages):
        # These recent messages should appear in full
        assert f"Message {i}:" in formatted


def test_format_history_pruning_with_limit_param(fresh_lead):
    """Verifies pruning works correctly when limit parameter is used."""
    # Create long messages
    long_content = "b" * 1000
    for i in range(15):
        msg = Message(
            lead_id=fresh_lead.lead_id,
            role=MessageRole.LEAD if i % 2 == 0 else MessageRole.ASSISTANT,
            content=f"Message {i}: {long_content}"
        )
        fresh_lead.add_message(msg)

    # Request only last 8 messages
    formatted = fresh_lead.format_history(limit=8)

    # Should only consider last 8 messages for pruning
    # Message 0-6 should not appear at all
    assert "Message 0:" not in formatted
    assert "Message 6:" not in formatted

    # Message 7-14 should be considered for pruning
    # Should still respect max_context_chars (with small buffer for summary lines)
    assert len(formatted) <= 6100


def test_format_history_empty_returns_empty_string(fresh_lead):
    """Verifies empty history returns empty string even with pruning logic."""
    formatted = fresh_lead.format_history()
    assert formatted == ""


def test_format_history_pruning_preserves_roles(fresh_lead):
    """Verifies role prefixes are preserved during pruning."""
    # Create long messages to trigger pruning
    long_content = "c" * 800
    for i in range(10):
        msg = Message(
            lead_id=fresh_lead.lead_id,
            role=MessageRole.LEAD if i % 2 == 0 else MessageRole.ASSISTANT,
            content=f"Message {i}: {long_content}"
        )
        fresh_lead.add_message(msg)

    formatted = fresh_lead.format_history(include_roles=True)

    # Should include role prefixes
    assert "LEAD:" in formatted
    assert "ASSISTANT:" in formatted


def test_format_history_pruning_without_roles(fresh_lead):
    """Verifies pruning works correctly without role prefixes."""
    # Create long messages
    long_content = "d" * 800
    for i in range(10):
        msg = Message(
            lead_id=fresh_lead.lead_id,
            role=MessageRole.LEAD if i % 2 == 0 else MessageRole.ASSISTANT,
            content=f"Message {i}: {long_content}"
        )
        fresh_lead.add_message(msg)

    formatted = fresh_lead.format_history(include_roles=False)

    # Should not include role prefixes
    assert "LEAD:" not in formatted
    assert "ASSISTANT:" not in formatted
    # Should still be pruned (with small buffer for summary lines)
    assert len(formatted) <= 6100