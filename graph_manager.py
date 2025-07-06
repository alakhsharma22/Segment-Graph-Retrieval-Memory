import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

class GraphManager:
    def __init__(self, k: int = 3):
        self.G = nx.DiGraph()
        self.k = k
        self.embeddings = {} 
        self.chunk_texts = {} 

    def add_node(self, chunk_id: str, embedding: np.ndarray, text: str):
        """
        Adds a new chunk as a node, links to previous node and top-k similar nodes.
        """
        self.G.add_node(chunk_id)
        self.embeddings[chunk_id] = embedding
        self.chunk_texts[chunk_id] = text

        nodes = list(self.G.nodes)
        if len(nodes) > 1:
            prev_id = nodes[-2]
            self.G.add_edge(prev_id, chunk_id)

        if len(self.embeddings) > 1:
            other_ids = [cid for cid in self.embeddings if cid != chunk_id]
            other_embs = np.vstack([self.embeddings[cid] for cid in other_ids])
            sims = cosine_similarity(embedding.reshape(1, -1), other_embs).flatten()
            topk = sims.argsort()[-self.k:][::-1]
            for idx in topk:
                similar_id = other_ids[idx]
                self.G.add_edge(chunk_id, similar_id)

    def retrieve_paths(self, seed_id: str, top_n: int = 3) -> List[str]:
        """
        Finds up to `top_n` shortest paths from `seed_id` to other nodes,
        then flattens and returns unique node IDs in order of navigation.
        """
        paths = []
        for node in self.G.nodes:
            if nx.has_path(self.G, seed_id, node):
                paths.append(nx.shortest_path(self.G, seed_id, node))
        paths.sort(key=len)

        result = []
        for path in paths[:top_n]:
            for cid in path:
                if cid not in result:
                    result.append(cid)
        return result
