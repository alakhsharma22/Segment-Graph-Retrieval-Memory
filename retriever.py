import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
from graph_manager import GraphManager
from embedder import Embedder

class Retriever:
    def __init__(self, graph_manager: GraphManager, embedder: Embedder):
        self.graph = graph_manager
        self.embedder = embedder

    def get_context(self, recent_text: str, top_n: int = 3) -> List[str]:
        """
        Embeds `recent_text`, finds the most similar chunk (seed),
        retrieves related chunks, and returns their raw texts.
        """
        query_emb = self.embedder.model.encode([recent_text])[0]
        all_ids = list(self.graph.embeddings.keys())
        all_embs = np.vstack([self.graph.embeddings[cid] for cid in all_ids])
        sims = cosine_similarity(query_emb.reshape(1, -1), all_embs).flatten()
        seed_idx = sims.argmax()
        seed_id = all_ids[seed_idx]

        selected_ids = self.graph.retrieve_paths(seed_id, top_n)
        return [self.graph.chunk_texts[cid] for cid in selected_ids]

