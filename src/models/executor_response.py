from pydantic import BaseModel, Field
from typing import Annotated

class OutboundMessage(BaseModel):
    content: str = Field(..., description="1-3 sentences for WhatsApp. Warm, human, Mexican-market friendly.")
    persona_reasoning: str = Field(..., description="Why this phrasing was chosen? e.g. Build trust, Be helpful, Nurturing, etc.")

class ExecutorResponse(BaseModel):
    """
    The Alena Gomez Output Contract.
    Standardized for Observability.
    """
    message: OutboundMessage
    
    # Traceability & Feedback Loop
    agreement_level: Annotated[float, Field(ge=0, le=1.0)] = Field(
        ..., description="How well does the Director's strategy fit the conversation context?"
    )
    feedback_for_director: str | None = Field(
        None, description="Observations for the Strategic Director regarding lead patterns."
    )
    
    # Summary for MongoDB
    execution_summary: str = Field(..., max_length=150)