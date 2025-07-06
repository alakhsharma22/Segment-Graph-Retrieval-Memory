from typing import List, Tuple
from transformers import PreTrainedTokenizer

class Chunker:
    def __init__(self, tokenizer: PreTrainedTokenizer, chunk_size: int = 512):
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> List[Tuple[str, str]]:
        """
        Splits `text` into chunks of up to `chunk_size` tokens.
        Returns a list of (chunk_id, chunk_text).
        """
        token_ids = self.tokenizer.encode(text)
        chunks = []
        for i in range(0, len(token_ids), self.chunk_size):
            chunk_ids = token_ids[i: i + self.chunk_size]
            chunk_text = self.tokenizer.decode(chunk_ids)
            chunk_id = f"chunk_{i // self.chunk_size}"
            chunks.append((chunk_id, chunk_text))
        return chunks
