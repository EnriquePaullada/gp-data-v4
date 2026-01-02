from enum import StrEnum
from typing import List, Optional
from pydantic import BaseModel, Field
from src.models.intelligence import BANTDimension 

class StrategicAction(StrEnum):
    HELP = "help"
    QUALIFY = "qualify"
    NURTURE = "nurture"
    CLOSE = "close"
    ESCALATE = "escalate"
    ABANDON = "abandon"

class MessageStrategy(BaseModel):
    tone: str = Field(..., description="e.g., empathetic, consultative, helpful, cheerful")
    language: str = Field(..., pattern="^(spanish|english)$")
    empathy_points: List[str] = Field(..., min_length=1)
    key_points: List[str] = Field(..., min_length=1)
    conversational_goal: str

class DirectorResponse(BaseModel):
    """The formal Strategic Command packet."""
    message_strategy: MessageStrategy
    
    # Deterministic Overrides
    action: StrategicAction
    suggested_stage_transition: Optional[str] = None
    focus_dimension: BANTDimension = BANTDimension.NEED
    # Scheduling context (if in progress)
    scheduling_instruction: Optional[str] = Field(
        None, description="Instruction for the scheduling engine (e.g. 'ask_city')"
    )
    
    strategic_reasoning: str = Field(..., max_length=600)