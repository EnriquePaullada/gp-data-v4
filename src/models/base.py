from datetime import datetime, timezone
from typing import Annotated, Optional
from pydantic import BaseModel, Field, ConfigDict, BeforeValidator

# Standardizes MongoDB ObjectIds to strings
PyObjectId = Annotated[str, BeforeValidator(str)]

class MongoBaseModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        # Standardize for React 19/Vite 7 frontend consumers
        json_encoders={datetime: lambda v: v.isoformat()},
        extra='forbid' 
    )

    id: Optional[PyObjectId] = Field(None, alias="_id")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))