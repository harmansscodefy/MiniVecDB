# minivectordb/api/schemas.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class VectorInsertSchema(BaseModel):
    id: str = Field(..., example="doc_1")
    vector: List[float] = Field(..., example=[0.1, 0.5, -0.2])
    metadata: Optional[Dict[str, Any]] = Field(default={}, example={"title": "Introduction to AI"})

class VectorSearchSchema(BaseModel):
    vector: List[float] = Field(..., example=[0.1, 0.5, -0.2])
    k: int = Field(default=5, ge=1, le=100)
    metric: str = Field(default="cosine", example="cosine")
    use_index: bool = Field(default=False)
    