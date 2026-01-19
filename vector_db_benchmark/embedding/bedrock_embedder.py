import asyncio
import json
from typing import Any

from vector_db_benchmark.utils.logging_config import log

from ..services.aws import AWSClient
from .base import EmbeddingProvider
from .config import EmbeddingConfig


class BedrockEmbedder(EmbeddingProvider):
    """AWS Bedrock embedding provider with concurrent processing."""

    def __init__(self, config: EmbeddingConfig, aws_client: AWSClient | None = None):
        """Initialize the Bedrock embedder."""
        super().__init__(config)
        self.aws_client = aws_client or AWSClient()

        log.info(f"Initialized Bedrock embedder: {self.config.model_name}")
        log.info(
            f"Configuration: {self.embedding_dimension}D, normalized={self.config.normalize}, max_concurrent={self.config.max_concurrent_requests}"
        )

    @property
    def tokenizer(self) -> Any | None:
        """Bedrock does not use a public tokenizer."""
        return None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embeds texts using concurrent Bedrock API calls."""
        if not texts:
            return []

        log.info(
            f"   ⚡ Embedding {len(texts)} texts concurrently (max {self.config.max_concurrent_requests} at once)"
        )

        return asyncio.run(self._embed_texts_async(texts))

    async def _embed_texts_async(self, texts: list[str]) -> list[list[float]]:
        """Embeds texts using async concurrent processing."""
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

        tasks = [self._embed_single_text_async(text, semaphore) for text in texts]

        embeddings = await asyncio.gather(*tasks, return_exceptions=True)

        successful_embeddings = []
        for i, result in enumerate(embeddings):
            if isinstance(result, Exception):
                log.warning(
                    f"Failed to embed text {i}: {result}, returning zero vector."
                )
                successful_embeddings.append([0.0] * self.embedding_dimension)
            else:
                successful_embeddings.append(result)

        return successful_embeddings

    async def _embed_single_text_async(
        self, text: str, semaphore: asyncio.Semaphore
    ) -> list[float]:
        """Embeds a single text using an async Bedrock API call."""
        async with semaphore:
            body = json.dumps(
                {
                    "inputText": text,
                    "dimensions": self.embedding_dimension,
                    "normalize": self.config.normalize,
                }
            )

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, self.aws_client.bedrock_embed, self.config.model_name, body
            )

            response_body = json.loads(response["body"].read())
            return response_body["embedding"]
