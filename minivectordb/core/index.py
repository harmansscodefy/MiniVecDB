import numpy as np
import random
from typing import List, Dict, Set, Tuple
from minivectordb.core.vector import calculate_similarity

class HNSWIndex:
    """
    Custom Hierarchical Navigable Small World (HNSW) Graph Index.
    """
    def __init__(self, dim: int, distance_metric: str = "cosine", m: int = 16, ef_construction: int = 64, mL: float = 1/np.log(16)):
        self.dim = dim
        self.metric = distance_metric
        self.m = m                       # Max edges per node
        self.ef_construction = ef_construction # Beam search width during insertion
        self.mL = mL                     # Multiplier for level generation
        
        self.entry_point: str = None
        self.max_level: int = -1
        
        # Graph structure: layers -> node_id -> list of neighbor node_ids
        self.graphs: List[Dict[str, List[str]]] = []
        # Local cache of vectors for indexing distance evaluations
        self.nodes: Dict[str, np.ndarray] = {}

    def _get_random_level(self) -> int:
        return int(-np.log(random.uniform(1e-5, 1.0)) * self.mL)

    def _dist(self, v1: np.ndarray, v2: np.ndarray) -> float:
        # Distance metric where SMALLER score is CLOSER
        sim = calculate_similarity(v1, v2, self.metric)
        return -sim if self.metric != "euclidean" else -sim  # similarity inverted to distance

    def add_node(self, node_id: str, vector: np.ndarray):
        self.nodes[node_id] = vector
        node_level = self._get_random_level()
        
        # Ensure graph layers exist up to node_level
        while len(self.graphs) <= node_level:
            self.graphs.append({})
            
        if self.entry_point is None:
            self.entry_point = node_id
            self.max_level = node_level
            for l in range(node_level + 1):
                self.graphs[l][node_id] = []
            return

        curr_obj = self.entry_point
        curr_dist = self._dist(vector, self.nodes[curr_obj])
        
        # Phase 1: Navigate top layers greedily down to node_level
        for l in range(self.max_level, node_level, -1):
            changed = True
            while changed:
                changed = False
                for neighbor in self.graphs[l].get(curr_obj, []):
                    d = self._dist(vector, self.nodes[neighbor])
                    if d < curr_dist:
                        curr_dist = d
                        curr_obj = neighbor
                        changed = True

        # Phase 2: Insert into layers from node_level down to 0
        for l in range(min(node_level, self.max_level), -1, -1):
            candidates = self._search_layer(vector, curr_obj, self.ef_construction, l)
            # Select m nearest neighbors
            neighbors = sorted(candidates, key=lambda x: x[1])[:self.m]
            
            self.graphs[l][node_id] = [n[0] for n in neighbors]
            
            # Add bidirectional edges
            for n_id, _ in neighbors:
                self.graphs[l][n_id].append(node_id)
                # Prune edges if max connections exceeded
                if len(self.graphs[l][n_id]) > self.m:
                    self.graphs[l][n_id] = self.graphs[l][n_id][:self.m]
                    
            if candidates:
                curr_obj = neighbors[0][0]

        if node_level > self.max_level:
            self.max_level = node_level
            self.entry_point = node_id

    def _search_layer(self, query: np.ndarray, entry_point: str, ef: int, level: int) -> List[Tuple[str, float]]:
        v = set([entry_point])
        C = [(entry_point, self._dist(query, self.nodes[entry_point]))] # Candidates
        W = list(C)                                                     # Dynamic result set

        while len(C) > 0:
            C.sort(key=lambda x: x[1])
            curr, curr_d = C.pop(0)
            
            W.sort(key=lambda x: x[1])
            farthest_w_dist = W[-1][1]
            
            if curr_d > farthest_w_dist and len(W) >= ef:
                break
                
            for neighbor in self.graphs[level].get(curr, []):
                if neighbor not in v:
                    v.add(neighbor)
                    d = self._dist(query, self.nodes[neighbor])
                    if d < farthest_w_dist or len(W) < ef:
                        C.append((neighbor, d))
                        W.append((neighbor, d))
                        W.sort(key=lambda x: x[1])
                        if len(W) > ef:
                            W.pop()
        return W

    def search(self, query: np.ndarray, k: int, ef_search: int = 32) -> List[Tuple[str, float]]:
        if self.entry_point is None:
            return []
            
        curr_obj = self.entry_point
        curr_dist = self._dist(query, self.nodes[curr_obj])
        
        # Traverse top layers greedily
        for l in range(self.max_level, 0, -1):
            changed = True
            while changed:
                changed = False
                for neighbor in self.graphs[l].get(curr_obj, []):
                    d = self._dist(query, self.nodes[neighbor])
                    if d < curr_dist:
                        curr_dist = d
                        curr_obj = neighbor
                        changed = True
                        
        # Search base layer (layer 0)
        candidates = self._search_layer(query, curr_obj, max(ef_search, k), level=0)
        candidates.sort(key=lambda x: x[1])
        
        # Convert distances back to similarity scores
        return [(n_id, -d) for n_id, d in candidates[:k]]

    def remove_node(self, node_id: str):
        if node_id in self.nodes:
            del self.nodes[node_id]
            for layer in self.graphs:
                if node_id in layer:
                    del layer[node_id]
                for n_id, neighbors in layer.items():
                    if node_id in neighbors:
                        neighbors.remove(node_id)
            if self.entry_point == node_id:
                self.entry_point = next(iter(self.nodes.keys())) if self.nodes else None