import numpy as np
from typing import List, Dict, Any, Optional
from minivectordb.core.vector import validate_vector, calculate_similarity
from minivectordb.core.index import HNSWIndex
from minivectordb.storage.persistence import StorageManager

class MiniVecDB:
    def __init__(self, dim: Optional[int] = None, storage_path: str = "minivecdb_data.json"):
        self.dim = dim
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.index: Optional[HNSWIndex] = None
        self.storage = StorageManager(storage_path)

    def insert(self, id: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None):
        v_arr = validate_vector(vector, expected_dim=self.dim)
        if self.dim is None:
            self.dim = len(v_arr)

        self.vectors[id] = v_arr
        self.metadata[id] = metadata or {}

        if self.index is not None:
            self.index.add_node(id, v_arr)

    def get(self, id: str) -> Optional[Dict[str, Any]]:
        if id not in self.vectors:
            return None
        return {
            "id": id,
            "vector": self.vectors[id].tolist(),
            "metadata": self.metadata[id]
        }

    def update(self, id: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None):
        if id not in self.vectors:
            raise KeyError(f"Vector ID '{id}' does not exist.")
        self.insert(id, vector, metadata)

    def delete(self, id: str) -> bool:
        if id not in self.vectors:
            return False
        del self.vectors[id]
        del self.metadata[id]
        if self.index is not None:
            self.index.remove_node(id)
        return True

    def count(self) -> int:
        return len(self.vectors)

    def build_index(self, metric: str = "cosine"):
        if self.dim is None:
            raise ValueError("Cannot build index on an empty database.")
        self.index = HNSWIndex(dim=self.dim, distance_metric=metric)
        for vec_id, vec in self.vectors.items():
            self.index.add_node(vec_id, vec)

    def search(self, query_vector: List[float], k: int = 5, metric: str = "cosine", use_index: bool = False) -> List[Dict[str, Any]]:
        q_arr = validate_vector(query_vector, expected_dim=self.dim)
        
        if use_index:
            if self.index is None:
                self.build_index(metric=metric)
            indexed_results = self.index.search(q_arr, k=k)
            results = []
            for vec_id, score in indexed_results:
                results.append({
                    "id": vec_id,
                    "score": round(float(score), 4),
                    "metadata": self.metadata.get(vec_id, {})
                })
            return results

        # Brute-force exact search
        scores = []
        for vec_id, vec in self.vectors.items():
            score = calculate_similarity(q_arr, vec, metric=metric)
            scores.append((vec_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_k = scores[:k]

        return [
            {
                "id": vec_id,
                "score": round(float(score), 4),
                "metadata": self.metadata.get(vec_id, {})
            }
            for vec_id, score in top_k
        ]

    def save(self):
        self.storage.save(self.vectors, self.metadata)

    def load(self):
        self.vectors, self.metadata = self.storage.load()
        if self.vectors:
            first_key = next(iter(self.vectors))
            self.dim = len(self.vectors[first_key])