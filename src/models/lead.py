import datetime as dt
from enum import StrEnum
from typing import List, Dict, Optional
from pydantic import Field, computed_field
from src.models.base import MongoBaseModel
from src.models.intelligence import IntelligenceSignal, BANTDimension
from src.models.message import Message

class SalesStage(StrEnum):
    NEW = "0 - initial_contact"
    DISCOVERY = "1 - discovery"
    QUALIFIED = "2 - qualified"
    DEMO_SCHEDULED = "3 - demo_scheduled"
    ON_HOLD = "4 - on_hold"
    WON = "5 - closed_won"
    LOST = "6 - closed_lost"

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