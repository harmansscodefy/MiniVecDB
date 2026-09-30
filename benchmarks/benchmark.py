import time
import numpy as np
from minivectordb.core.database import MiniVecDB

def run_benchmark(num_vectors: int = 5000, dim: int = 64, k: int = 10):
    print(f"\n--- Running Benchmark: {num_vectors} Vectors | Dim: {dim} ---")
    db = MiniVecDB(dim=dim)
    
    # Generate random test dataset
    np.random.seed(42)
    data = np.random.randn(num_vectors, dim).astype(np.float32)
    for i, vec in enumerate(data):
        db.insert(f"vec_{i}", vec.tolist())
        
    query = np.random.randn(dim).tolist()
    
    # 1. Benchmark Brute Force
    start = time.perf_counter()
    exact_results = db.search(query, k=k, use_index=False)
    bf_time = (time.perf_counter() - start) * 1000
    
    # 2. Benchmark HNSW Index
    print("Building HNSW Index...")
    db.build_index()
    
    start = time.perf_counter()
    hnsw_results = db.search(query, k=k, use_index=True)
    hnsw_time = (time.perf_counter() - start) * 1000
    
    # Calculate Recall@K
    exact_ids = set([r["id"] for r in exact_results])
    hnsw_ids = set([r["id"] for r in hnsw_results])
    recall = len(exact_ids.intersection(hnsw_ids)) / k
    
    speedup = bf_time / hnsw_time if hnsw_time > 0 else 0
    
    print(f"Brute-Force Latency : {bf_time:.3f} ms")
    print(f"HNSW Search Latency : {hnsw_time:.3f} ms")
    print(f"Speedup Multiplier  : {speedup:.2f}x faster")
    print(f"Recall@{k}             : {recall * 100:.1f}%")

if __name__ == "__main__":
    run_benchmark(num_vectors=2000)
