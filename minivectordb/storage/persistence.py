import json
import os
import numpy as np
from typing import Dict, Any, Tuple

class StorageManager:
    """
    Handles persistence of vectors and metadata to disk using JSON format.
    """
    def __init__(self, filepath: str = "minivecdb_data.json"):
        self.filepath = filepath

    def save(self, vectors: Dict[str, np.ndarray], metadata: Dict[str, Dict[str, Any]]) -> bool:
        data = {
            "vectors": {k: v.tolist() for k, v in vectors.items()},
            "metadata": metadata
        }
        temp_filepath = f"{self.filepath}.tmp"
        try:
            with open(temp_filepath, 'w') as f:
                json.dump(data, f, indent=2)
            os.replace(temp_filepath, self.filepath) # Atomic write
            return True
        except Exception as e:
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            raise IOError(f"Failed to save database: {str(e)}")

    def load(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        if not os.path.exists(self.filepath):
            return {}, {}
            
        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)
            
            vectors = {k: np.array(v, dtype=np.float32) for k, v in data.get("vectors", {}).items()}
            metadata = data.get("metadata", {})
            return vectors, metadata
        except Exception as e:
            raise IOError(f"Corrupted database file or read error: {str(e)}")