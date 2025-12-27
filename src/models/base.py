import datetime as dt
from typing import Annotated, Optional
from pydantic import BaseModel, Field, ConfigDict, BeforeValidator, field_serializer

# Standardizes MongoDB ObjectIds to strings
PyObjectId = Annotated[str, BeforeValidator(str)]

class MongoBaseModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        extra='forbid' 
    )

    id: Optional[PyObjectId] = Field(None, alias="_id")
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))
    updated_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))

    # MODERN SERIALIZATION: This is the Late-2025 way
    @field_serializer("created_at", "updated_at")
    def serialize_dt(self, value: dt.datetime):
        return value.isoformat()