import re
from abc import ABC, abstractmethod
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from vector_db_benchmark.utils.logging_config import log

from ..embedding.config import EmbeddingConfig


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    def __init__(self, config: EmbeddingConfig):
        """Initialize the embedding provider."""
        self.config = config
        self.model_name = config.model_name
        self._text_splitter = None
        self.chunking_strategy = config.chunking_strategy
        self._setup_chunking()

    def _setup_chunking(self):
        """Sets up text chunking based on the strategy defined in the config."""
        strategy = self.chunking_strategy
        if strategy.chunking_type == "token":
            if self.tokenizer is None:
                raise NotImplementedError(
                    "Tokenizer must be implemented for token-based chunking"
                )
            self._text_splitter = (
                RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                    self.tokenizer,
                    chunk_size=strategy.chunk_size,
                    chunk_overlap=strategy.chunk_overlap,
                )
            )
        else:
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=strategy.chunk_size,
                chunk_overlap=strategy.chunk_overlap,
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            )

    @property
    def text_splitter(self):
        """Lazy-loads the text splitter."""
        if self._text_splitter is None:
            self._setup_chunking()
        return self._text_splitter

    @property
    @abstractmethod
    def tokenizer(self) -> Any | None:
        """Returns the tokenizer for this provider, if applicable."""
        return None

    @property
    def embedding_dimension(self) -> int:
        """Gets the embedding dimension from the configuration."""
        if self.config.dimensions is None:
            raise ValueError(
                f"Embedding dimension not configured for model {self.model_name}"
            )
        return self.config.dimensions

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embeds a list of texts into vectors."""

    def embed_document(
        self, text: str, metadata: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Embeds a document with an optimal chunking strategy."""
        cleaned_text = self._clean_text(text)
        text_chunks = self.text_splitter.split_text(cleaned_text)

        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk_dict = {
                "text": chunk_text,
                "chunk_id": i,
                "chunk_size": len(chunk_text),
                **metadata,
            }
            chunks.append(chunk_dict)

        chunk_texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embed_texts(chunk_texts)

        for chunk, embedding in zip(chunks, embeddings, strict=True):
            chunk["embedding"] = embedding

        log.info(f"   ✓ Created {len(chunks)} chunks for {self.__class__.__name__}")
        return chunks

    def _clean_text(self, text: str) -> str:
        """Cleans text for processing."""
        text = re.sub(r"\s+", " ", text)
        return text.strip()
