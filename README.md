cat << 'EOF' > README.md
# MiniVecDB 🚀

A functional, high-performance, lightweight vector database built from scratch in Python. Designed as a modular portfolio project demonstrating custom Approximate Nearest Neighbor (ANN) indexing using Hierarchical Navigable Small World (HNSW) graphs, atomic disk persistence, vector similarity math, and RESTful API serving.

---

## 📌 Architecture & Folder Structure

MiniVecDB separates concerns into discrete, modular layers:

```text
minivectordb/
│
├── minivectordb/                   # Core Python Package
│   ├── __init__.py                 # Version exposure and package entry
│   ├── core/                       # Core Database & Search Engine
│   │   ├── __init__.py
│   │   ├── database.py             # CRUD operations & vector search routing
│   │   ├── vector.py               # Vector math (Cosine, Euclidean, Dot Product)
│   │   └── index.py                # Custom HNSW graph index implementation
│   │
│   ├── storage/                    # Storage & Persistence Layer
│   │   ├── __init__.py
│   │   └── persistence.py          # Atomic JSON file I/O operations
│   │
│   └── api/                        # REST API Layer
│       ├── __init__.py
│       └── routes.py               # FastAPI endpoints & Pydantic validation
│
├── tests/                          # Automated Pytest Suite
│   ├── test_vector.py              # Unit tests for vector operations & math
│   ├── test_database.py            # Unit tests for database CRUD
│   └── test_index.py               # Tests for HNSW index building & search
│
├── benchmarks/                     # Performance Benchmarks
│   └── benchmark.py                # Benchmark measuring Latency, QPS, & Recall@K
│
├── examples/                       # Demonstrations & Tutorials
│   └── rag_demo.py                 # End-to-end RAG pipeline integration
│
├── conftest.py                     # Pytest path resolution configuration
├── pyproject.toml                  # Python package configuration
├── requirements.txt                # System dependencies
└── README.md                       # Comprehensive documentation

🔄 System Workflows
1. Data Ingestion Workflow
Plaintext
[ Raw Vector + Metadata ]
           │
           ▼
 [ Vector Validation ] ────► Validates vector dimensions (e.g., dim=64) & data types
           │
           ▼
   [ Core Storage ]     ────► Stores normalized float32 array in memory & dict metadata
           │
           ▼
[ HNSW Index Builder ]  ────► Inserts vector into probabilistic multi-layer graph
2. Query & Search Workflow
Plaintext
[ Incoming Query Vector ]
           │
           ▼
 [ Dimensionality Check ]
           │
           ├───► If use_index=False ──► [ Brute-Force Scan ] ──► Computes distance vs ALL N vectors
           │                                                                 │
           └───► If use_index=True  ──► [ HNSW Graph Search ] ───────────────┤
                                        • Traverses top sparse layer         │
                                        • Greedy search to nearest node      │
                                        • Steps down layers to dense graph   │
                                                                             ▼
                                                             [ Ranked Top-K Results ]
3. Atomic Disk Persistence Workflow
Plaintext
[ In-Memory Database State ]
               │
               ▼
   [ Serialize to JSON Format ]
               │
               ▼
 [ Write to Temporary File (.tmp) ] ──► Ensures full write completes without corruption
               │
               ▼
  [ Atomic OS Replace Operation ]  ──► `os.replace(".tmp", "db.json")` safely updates state
🚀 Quickstart Guide
1. Installation
Clone the repository and install requirements:
Bash
git clone [https://github.com/harmansscodefy/MiniVecDB.git](https://github.com/harmansscodefy/MiniVecDB.git)
cd MiniVecDB
pip install -r requirements.txt
2. Python SDK Usage
Python
from minivectordb.core.database import MiniVecDB

# Initialize database instance (e.g., 3-dimensional vectors)
db = MiniVecDB(dim=3)

# Insert vectors with metadata payloads
db.insert("doc_1", [0.1, 0.8, 0.1], metadata={"title": "Machine Learning"})
db.insert("doc_2", [0.9, 0.1, 0.0], metadata={"title": "Database Systems"})

# Construct the HNSW index for sub-linear search
db.build_index(metric="cosine")

# Query nearest neighbors using approximate search
results = db.search([0.15, 0.75, 0.1], k=1, use_index=True)
print(results)
3. Running the REST API
Start the FastAPI application server using Uvicorn:
Bash
python3 -m uvicorn minivectordb.api.routes:app --reload
Access the interactive API documentation (Swagger UI) at http://127.0.0.1:8000/docs.
📊 Benchmarks & Performance
Benchmark results evaluated on synthetic normalized float32 embeddings (N=2,000, D=64, K=10):
Search Method	Search Latency	Throughput (QPS)	Recall@10	Performance Gain
Brute Force (Exact)	~3.82 ms	~260 QPS	100.0%	1.0x (Baseline)
HNSW Index (ANN)	~0.35 ms	~2,850 QPS	90.0%	~10.9x Speedup
Run the benchmark suite locally:
Bash
python3 benchmarks/benchmark.py
🧪 Automated Testing
Execute the unit test suite covering vector math, database CRUD, and HNSW graph accuracy:
Bash
python3 -m pytest