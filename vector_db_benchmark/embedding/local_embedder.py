from typing import Any

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from vector_db_benchmark.utils.logging_config import log

from .base import EmbeddingProvider
from .config import EmbeddingConfig


class LocalEmbedder(EmbeddingProvider):
    """Local sentence-transformers embedding provider."""

    def __init__(self, config: EmbeddingConfig):
        """Initializes the local embedder."""
        super().__init__(config)

        log.info(f"Loading embedding model: {self.config.model_name}")
        self.model = SentenceTransformer(self.config.model_name)
        self._tokenizer = None

        log.info(f"Model loaded. Embedding dimension: {self.embedding_dimension}")

    @property
    def tokenizer(self) -> Any:
        """Lazy-loads tokenizer on first access."""
        if self._tokenizer is None:
            log.info(f"Initializing tokenizer for {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generates embeddings for a batch of texts."""
        embeddings = self.model.encode(
            texts, convert_to_tensor=False, show_progress_bar=True
        )
        return embeddings.tolist()
