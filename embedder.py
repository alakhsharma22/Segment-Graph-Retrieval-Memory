import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, chunks: List[Tuple[str, str]]) -> Dict[str, np.ndarray]:
        """
        Takes a list of (chunk_id, chunk_text) and returns a mapping
        chunk_id -> embedding vector.
        """
        texts = [text for _, text in chunks]
        embeddings = self.model.encode(texts)
        return {chunk_id: emb for (chunk_id, _), emb in zip(chunks, embeddings)}
