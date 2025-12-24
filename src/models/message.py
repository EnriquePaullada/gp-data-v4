import datetime as dt
from enum import StrEnum
from pydantic import Field
from src.models.base import MongoBaseModel


class MessageRole(StrEnum):
    LEAD = "lead"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    HUMAN = "human"

class Message(MongoBaseModel):
    """"
    The Atomic Interaction Model.
    Inherits created_at from MongoBaseModel, but uses 'timestamp' 
    for domain-specific clarity in conversation flows.
    """
    lead_id: str = Field(..., index=True) # Linked to Lead.lead_id
    role: MessageRole
    content: str
    # Store tokens used for this specific message for cost tracking
    tokens: int = 0
    timestamp: dt.datetime = Field(
        default_factory=lambda: dt.datetime.now(dt.UTC),
        description="The exact moment the message was sent or received."
    )