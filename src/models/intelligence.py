from enum import StrEnum
from typing import Annotated, List, Optional
from pydantic import BaseModel, Field

class BANTDimension(StrEnum):
    BUDGET = "budget"
    AUTHORITY = "authority"
    NEED = "need"
    TIMELINE = "timeline"

class ConfidenceScore(BaseModel):
    """Encapsulates AI certainty with mandatory justification."""
    value: Annotated[float, Field(ge=0, le=1.0)]
    reasoning: str = Field(..., max_length=300)

class IntelligenceSignal(BaseModel):
    """A single atomic fact extracted from conversation history."""
    dimension: BANTDimension
    extracted_value: str
    confidence: ConfidenceScore
    source_message_id: str = Field(..., description="The ID of the message that provided this signal.")
    is_inferred: bool = Field(default=False)
    inferred_from: list[str] | None = Field(
        default=None, 
        description="List of message IDs used to synthesize this intelligence."
    )
    raw_evidence: str = Field(..., description="Verbatim snippet from the lead.")