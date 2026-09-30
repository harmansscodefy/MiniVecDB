import time
import numpy as np
from typing import List, Dict, Any
from minivectordb.core.database import MiniVecDB

def run_benchmark(num_vectors: int = 5000, dim: int = 64, k: int = 10):
    print("=" * 80)
    print(f"                      MINIVECTORDB PERFORMANCE BENCHMARK            ")
    print("=" * 80)
    print(f"Dataset Size : {num_vectors:,} vectors")
    print(f"Vector Dim   : {dim} dimensions")
    print(f"Top-K        : K = {k}")
    print(f"Metric       : Cosine Similarity")
    print("-" * 80)

    # Initialize MiniVecDB
    db = MiniVecDB(dim=dim)

    # Generate synthetic vector dataset
    print(f"[1/4] Generating {num_vectors:,} synthetic vectors...")
    np.random.seed(42)
    raw_vectors = np.random.randn(num_vectors, dim).astype(np.float32)
    
    # Normalize vectors for realistic embedding distribution
    norms = np.linalg.norm(raw_vectors, axis=1, keepdims=True)
    raw_vectors = raw_vectors / np.maximum(norms, 1e-12)

    # Ingest vectors into database
    print("[2/4] Ingesting vectors into MiniVecDB storage...")
    for i, vec in enumerate(raw_vectors):
        db.insert(f"vec_{i}", vec.tolist(), metadata={"index": i})

    # Prepare query vector
    query_vector = np.random.randn(dim).astype(np.float32)
    query_norm = np.linalg.norm(query_vector)
    if query_norm > 0:
        query_vector = (query_vector / query_norm).tolist()
    else:
        query_vector = query_vector.tolist()

    # 1. Measure Exact Brute-Force Search
    print("[3/4] Running Exact Brute-Force Search...")
    bf_start = time.perf_counter()
    exact_results = db.search(query_vector, k=k, metric="cosine", use_index=False)
    bf_latency_ms = (time.perf_counter() - bf_start) * 1000.0
    bf_qps = 1000.0 / bf_latency_ms if bf_latency_ms > 0 else 0

    # 2. Build HNSW Index and Measure Search
    print("[4/4] Building HNSW Index & Running Approximate Search...")
    build_start = time.perf_counter()
    db.build_index(metric="cosine")
    build_time_ms = (time.perf_counter() - build_start) * 1000.0

    hnsw_start = time.perf_counter()
    hnsw_results = db.search(query_vector, k=k, metric="cosine", use_index=True)
    hnsw_latency_ms = (time.perf_counter() - hnsw_start) * 1000.0
    hnsw_qps = 1000.0 / hnsw_latency_ms if hnsw_latency_ms > 0 else 0

    # Calculate Recall@K
    exact_ids = set([res["id"] for res in exact_results])
    hnsw_ids = set([res["id"] for res in hnsw_results])
    intersection = exact_ids.intersection(hnsw_ids)
    recall_at_k = (len(intersection) / k) * 100.0

    speedup = bf_latency_ms / hnsw_latency_ms if hnsw_latency_ms > 0 else 1.0

    # Output Structured Performance Table
    print("\n" + "=" * 80)
    print("                                BENCHMARK RESULTS                           ")
    print("=" * 80)
    print(f"{'Metric / Method':<22} | {'Brute-Force (Exact)':<22} | {'HNSW Index (ANN)':<22}")
    print("-" * 80)
    print(f"{'Search Latency':<22} | {bf_latency_ms:>18.3f} ms | {hnsw_latency_ms:>18.3f} ms")
    print(f"{'Throughput (QPS)':<22} | {bf_qps:>18.2f} QPS | {hnsw_qps:>18.2f} QPS")
    print(f"{'Recall@' + str(k):<22} | {'100.0%':>22} | {f'{recall_at_k:.1f}%':>22}")
    print(f"{'Search Speedup':<22} | {'1.0x (Baseline)':>22} | {f'{speedup:.2f}x Faster':>22}")
    print("-" * 80)
    print(f"HNSW Index Construction Time: {build_time_ms:.2f} ms")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    run_benchmark(num_vectors=2000, dim=64, k=10)
