import numpy as np
from typing import List, Union

VectorInput = Union[List[float], np.ndarray]

def validate_vector(v: VectorInput, expected_dim: int = None) -> np.ndarray:
    if isinstance(v, list):
        v = np.array(v, dtype=np.float32)
    elif not isinstance(v, np.ndarray):
        raise TypeError("Vector must be a list or numpy array")
    
    if v.ndim != 1:
        raise ValueError("Vector must be 1-dimensional")
        
    if expected_dim is not None and len(v) != expected_dim:
        raise ValueError(f"Vector dimension mismatch: expected {expected_dim}, got {len(v)}")
        
    return v.astype(np.float32)

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))

def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.linalg.norm(v1 - v2))

def dot_product(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.dot(v1, v2))

def calculate_similarity(v1: VectorInput, v2: VectorInput, metric: str = "cosine") -> float:
    v1_arr = validate_vector(v1)
    v2_arr = validate_vector(v2, expected_dim=len(v1_arr))
    
    metric = metric.lower()
    if metric == "cosine":
        return cosine_similarity(v1_arr, v2_arr)
    elif metric == "euclidean":
        # Negate distance so higher values mean "more similar" for Top-K ranking
        return -euclidean_distance(v1_arr, v2_arr)
    elif metric == "dot":
        return dot_product(v1_arr, v2_arr)
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'cosine', 'euclidean', or 'dot'")