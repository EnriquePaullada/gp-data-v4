from enum import StrEnum
from typing import List, Literal
from pydantic import BaseModel, Field
from src.models.intelligence import IntelligenceSignal, Sentiment

class Intent(StrEnum):
    GREETING = "greeting"
    FOLLOWUP = "followup_response"
    PRICING = "pricing_question"
    DEMO = "demo_request"
    FEATURE = "feature_question"
    TECHNICAL = "technical_question"
    OBJECTION = "objection"
    READY_TO_BUY = "ready_to_buy"
    SUPPORT = "support_request"
    GENERAL = "general_inquiry"
    UNCLEAR = "unclear"

class UrgencyLevel(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCLEAR = "unclear"

class ClassifierResponse(BaseModel):
    """The formal output contract for the Classifier Agent."""
    intent: Intent
    intent_confidence: float = Field(ge=0, le=1.0)
    
    topic: str = Field(..., description="The primary subject of the message.")
    topic_confidence: float = Field(ge=0, le=1.0)
    
    urgency: UrgencyLevel
    urgency_confidence: float = Field(ge=0, le=1.0)
    
    language: Literal["spanish", "english", "mixed"]
    sentiment: Sentiment
    engagement_level: Literal["high", "medium", "low"]
    
    requires_human_escalation: bool
    reasoning: str = Field(..., max_length=600)
    
    # Instead of a raw dict, we return a list of our typed signals
    new_signals: List[IntelligenceSignal] = Field(default_factory=list)