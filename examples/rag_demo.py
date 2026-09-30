import numpy as np
from minivectordb.core.database import MiniVecDB

def simple_embedder(text: str) -> list:
    vocab = ["ai", "machine", "learning", "database", "vector", "sql", "data"]
    tokens = text.lower().split()
    vec = [float(tokens.count(w)) for w in vocab]
    norm = np.linalg.norm(vec)
    return (np.array(vec) / norm if norm > 0 else np.array(vec)).tolist()

def main():
    db = MiniVecDB(dim=7)
    
    documents = [
        {"id": "doc1", "text": "AI and machine learning powering modern search."},
        {"id": "doc2", "text": "Relational database uses SQL for querying data."},
        {"id": "doc3", "text": "Vector database indexes mathematical embeddings for vector search."}
    ]
    
    for doc in documents:
        vec = simple_embedder(doc["text"])
        db.insert(doc["id"], vec, metadata={"text": doc["text"]})
        
    query = "Tell me about machine learning and vector search"
    q_vec = simple_embedder(query)
    
    results = db.search(q_vec, k=2)
    
    print(f"\nUser Query: '{query}'")
    print("Top Retrived Documents from MiniVecDB:")
    for r in results:
        print(f" - Score {r['score']}: {r['metadata']['text']}")

if __name__ == "__main__":
    main()
