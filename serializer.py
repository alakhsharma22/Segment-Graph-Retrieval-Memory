from typing import List
from transformers import pipeline

class Serializer:
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        self.summarizer = pipeline("summarization", model=model_name)

    def summarize(self, texts: List[str]) -> List[str]:
        """
        Returns a list of summary strings for each input text.
        """
        summaries = self.summarizer(texts, truncation=True)
        return [s["summary_text"] for s in summaries]
