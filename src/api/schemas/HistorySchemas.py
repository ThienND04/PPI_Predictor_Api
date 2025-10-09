from pydantic import BaseModel, Field
from typing import List, Optional, Union

class PredictionRecordOut(BaseModel):
    id: Union[int, str]
    model_name: str = Field(...)
    protein1_id: str = Field(...)
    protein2_id: str = Field(...)
    score: Optional[float] = None
    label: str = Field(...)
    timestamp: Optional[str] = Field(default=None, description="ISO8601 UTC time, e.g. 2025-10-10T08:22:00Z")

class HistoryResponse(BaseModel):
    user_id: Union[int, str]
    total_records: int
    predictions: List[PredictionRecordOut] = Field(default_factory=list)
