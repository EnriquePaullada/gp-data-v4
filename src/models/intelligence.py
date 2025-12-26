# src/models/intelligence.py
from enum import StrEnum
from typing import Annotated
from pydantic import BaseModel, Field

class BANTDimension(StrEnum):
    BUDGET = "budget"
    AUTHORITY = "authority"
    NEED = "need"
    TIMELINE = "timeline"

class Sentiment(StrEnum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    SKEPTICAL = "skeptical"
    ENTHUSIASTIC = "enthusiastic"
    UNCLEAR = "unclear"

class ConfidenceScore(BaseModel):
    value: Annotated[float, Field(ge=0, le=1.0)]
    reasoning: str = Field(..., max_length=300)

class IntelligenceSignal(BaseModel):
    """A high-precision atomic fact extracted from an interaction."""
    dimension: BANTDimension
    extracted_value: str
    confidence: ConfidenceScore
    source_message_id: str = Field(..., description="The ID of the primary message.")
    is_inferred: bool = Field(default=False)
    inferred_from: list[str] | None = Field(default=None)
    raw_evidence: str = Field(..., description="Verbatim snippet from the lead.")