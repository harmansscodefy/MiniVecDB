import pytest
import numpy as np
from minivectordb.core.database import MiniVecDB
from minivectordb.core.vector import calculate_similarity

def test_insert_and_get():
    db = MiniVecDB(dim=3)
    db.insert("v1", [1.0, 0.0, 0.0], {"name": "x-axis"})
    assert db.count() == 1
    
    item = db.get("v1")
    assert item["id"] == "v1"
    assert item["vector"] == [1.0, 0.0, 0.0]
    assert item["metadata"]["name"] == "x-axis"

def test_similarity_metrics():
    v1 = [1.0, 0.0]
    v2 = [0.0, 1.0]
    
    assert calculate_similarity(v1, v1, "cosine") == 1.0
    assert calculate_similarity(v1, v2, "cosine") == 0.0
    assert calculate_similarity(v1, v1, "dot") == 1.0

def test_brute_force_search():
    db = MiniVecDB(dim=2)
    db.insert("v1", [1.0, 0.0])
    db.insert("v2", [0.8, 0.2])
    db.insert("v3", [0.0, 1.0])
    
    results = db.search([1.0, 0.1], k=2, metric="cosine")
    assert len(results) == 2
    assert results[0]["id"] == "v1"
    assert results[1]["id"] == "v2"

def test_hnsw_index_search():
    db = MiniVecDB(dim=2)
    db.insert("v1", [1.0, 0.0])
    db.insert("v2", [0.9, 0.1])
    db.insert("v3", [0.0, 1.0])
    db.build_index()
    
    results = db.search([1.0, 0.0], k=1, use_index=True)
    assert len(results) == 1
    assert results[0]["id"] == "v1"