from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from minivectordb.core.database import MiniVecDB

# Initialize FastAPI application instance
app = FastAPI(
    title="MiniVecDB API",
    description="Lightweight Vector Database REST API",
    version="1.0.0"
)

# Global database instance initialized with default dimension 64
db = MiniVecDB(dim=64)

# Pydantic Request Schemas
class InsertRequest(BaseModel):
    id: str
    vector: List[float]
    metadata: Optional[Dict[str, Any]] = None

class SearchRequest(BaseModel):
    vector: List[float]
    k: int = 10
    metric: str = "cosine"
    use_index: bool = True

@app.get("/health")
def health_check():
    return {"status": "ok", "total_vectors": db.count()}

@app.post("/insert")
def insert_vector(payload: InsertRequest):
    try:
        db.insert(payload.id, payload.vector, payload.metadata)
        return {"message": "Vector inserted successfully", "id": payload.id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/search")
def search_vectors(payload: SearchRequest):
    try:
        results = db.search(
            vector=payload.vector,
            k=payload.k,
            metric=payload.metric,
            use_index=payload.use_index
        )
        return {"results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/build-index")
def build_index(metric: str = "cosine"):
    try:
        db.build_index(metric=metric)
        return {"message": "HNSW index built successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
