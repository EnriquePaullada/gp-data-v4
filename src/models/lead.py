import datetime as dt
from enum import StrEnum
from typing import List, Dict, Optional
from pydantic import Field, computed_field
from src.models.base import MongoBaseModel
from src.models.intelligence import IntelligenceSignal, BANTDimension
from src.models.message import Message
from src.config import get_settings

class SalesStage(StrEnum):
    NEW = "0 - initial_contact"
    DISCOVERY = "1 - discovery"
    QUALIFIED = "2 - qualified"
    DEMO_SCHEDULED = "3 - demo_scheduled"
    ON_HOLD = "4 - on_hold"
    WON = "5 - closed_won"
    LOST = "6 - closed_lost"


class HandoffStatus(StrEnum):
    """Human handoff states for escalation workflow."""
    NONE = "none"              # AI handling conversation
    REQUESTED = "requested"    # Escalation triggered, awaiting human
    ASSIGNED = "assigned"      # Human agent has taken over
    RESOLVED = "resolved"      # Handoff complete, AI can resume

class Lead(MongoBaseModel):
    """
    The Central Domain Model.
    Designed for Event-Sourced Intelligence.
    """
    lead_id: str = Field(..., description="E.164 formatted phone number")
    full_name: Optional[str] = None
    current_stage: SalesStage = SalesStage.NEW
    
    # WORKING MEMORY: 
    # We only keep the last 20 messages here for instant agent access.
    # The full history is fetched from the 'messages' collection only when needed.
    recent_history: List[Message] = Field(default_factory=list, max_length=20)

    # The 'Signals' are the event log of everything the AI has ever learned
    signals: List[IntelligenceSignal] = Field(default_factory=list)
    
    message_count: int = 0

    # Scheduling & Orchestration
    next_followup_at: Optional[dt.datetime] = None
    last_interaction_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))

    # Human Handoff
    handoff_status: HandoffStatus = HandoffStatus.NONE
    handoff_requested_at: Optional[dt.datetime] = None
    handoff_reason: Optional[str] = None
    handoff_assigned_to: Optional[str] = None  # Human agent identifier

    @computed_field
    @property
    def bant_summary(self) -> Dict[BANTDimension, str]:
        """
        DETERMINISTIC REDUCTION:
        Collapses the history of signals into the current 'Best Known Truth'.
        """
        summary = {dim: "unknown" for dim in BANTDimension}
        # Iterate through signals; later signals for the same dimension overwrite earlier ones
        for signal in self.signals:
            summary[signal.dimension] = signal.extracted_value
        return summary

    @computed_field
    @property
    def is_handed_off(self) -> bool:
        """Returns True if conversation is currently in human handoff state."""
        return self.handoff_status in (HandoffStatus.REQUESTED, HandoffStatus.ASSIGNED)

    def request_handoff(self, reason: str) -> None:
        """
        Initiate human handoff for this lead.

        Args:
            reason: Why the AI is escalating (e.g., "Complex technical question")
        """
        self.handoff_status = HandoffStatus.REQUESTED
        self.handoff_requested_at = dt.datetime.now(dt.UTC)
        self.handoff_reason = reason
        self.updated_at = dt.datetime.now(dt.UTC)

    def assign_handoff(self, agent_id: str) -> None:
        """
        Assign handoff to a human agent.

        Args:
            agent_id: Identifier for the human agent taking over
        """
        self.handoff_status = HandoffStatus.ASSIGNED
        self.handoff_assigned_to = agent_id
        self.updated_at = dt.datetime.now(dt.UTC)

    def resolve_handoff(self) -> None:
        """Mark handoff as resolved, allowing AI to resume."""
        self.handoff_status = HandoffStatus.RESOLVED
        self.updated_at = dt.datetime.now(dt.UTC)

    def clear_handoff(self) -> None:
        """Reset handoff state completely (AI takes over again)."""
        self.handoff_status = HandoffStatus.NONE
        self.handoff_requested_at = None
        self.handoff_reason = None
        self.handoff_assigned_to = None
        self.updated_at = dt.datetime.now(dt.UTC)

    def add_signal(self, signal: IntelligenceSignal):
        """Standard method for evolving lead state with a timestamp update."""
        self.signals.append(signal)
        self.updated_at = dt.datetime.now(dt.UTC)

    def add_message(self, message: Message):
        """Adds a message to the working memory window."""
        self.recent_history.append(message)
        # Maintain a sliding window (e.g., keep only last 20)
        if len(self.recent_history) > 20:
            self.recent_history.pop(0)

        self.message_count += 1
        self.last_interaction_at = message.timestamp if message.timestamp else dt.datetime.now(dt.UTC)

    def format_history(self, limit: int | None = None, include_roles: bool = True) -> str:
        """
        Formats conversation history for LLM context with smart pruning.

        Centralizes the repeated pattern of formatting messages across all agents.
        This eliminates DRY violations and ensures consistent formatting.

        Implements context pruning to prevent exceeding model context windows:
        - Always keeps the most recent messages in full
        - Truncates older messages if total exceeds max_context_chars
        - Provides summary when pruning occurs

        Args:
            limit: Number of recent messages to include (None = all in recent_history)
            include_roles: Whether to prefix each line with "ROLE:" (default: True)

        Returns:
            Formatted string ready for LLM prompt injection

        Example:
            >>> lead.format_history(limit=3)
            'LEAD: Hello\\nASSISTANT: Hi there!\\nLEAD: I need help'
        """
        settings = get_settings()
        messages = self.recent_history[-limit:] if limit else self.recent_history

        if not messages:
            return ""

        # Format all messages first
        formatted_lines = []
        if include_roles:
            formatted_lines = [f"{msg.role.upper()}: {msg.content}" for msg in messages]
        else:
            formatted_lines = [msg.content for msg in messages]

        # Join and check total length
        full_history = "\n".join(formatted_lines)

        # If within limit, return as-is
        if len(full_history) <= settings.max_context_chars:
            return full_history

        # Context pruning: keep recent messages in full, truncate older ones
        min_recent = min(settings.min_recent_messages, len(messages))
        recent_messages = messages[-min_recent:]
        older_messages = messages[:-min_recent]

        # Format recent messages (always kept in full)
        if include_roles:
            recent_lines = [f"{msg.role.upper()}: {msg.content}" for msg in recent_messages]
        else:
            recent_lines = [msg.content for msg in recent_messages]

        recent_context = "\n".join(recent_lines)
        remaining_chars = settings.max_context_chars - len(recent_context)

        # Calculate space for older messages
        if remaining_chars > 100 and older_messages:
            # Format older messages with truncation
            older_lines = []
            summary_line = f"[Earlier conversation: {len(older_messages)} messages truncated for context limit]"
            remaining_chars -= len(summary_line) + 2  # +2 for newlines

            # Try to fit as many older messages as possible
            chars_per_message = remaining_chars // len(older_messages)

            for msg in older_messages:
                if include_roles:
                    content = f"{msg.role.upper()}: {msg.content}"
                else:
                    content = msg.content

                if len(content) <= chars_per_message:
                    older_lines.append(content)
                else:
                    # Truncate with ellipsis
                    truncated = content[:chars_per_message - 3] + "..."
                    older_lines.append(truncated)

            # Combine: summary + truncated older + full recent
            return f"{summary_line}\n" + "\n".join(older_lines) + "\n" + recent_context
        else:
            # Not enough space for older messages, just return recent with summary
            pruned_count = len(older_messages)
            summary = f"[{pruned_count} earlier messages omitted due to context limit]\n"
            return summary + recent_context