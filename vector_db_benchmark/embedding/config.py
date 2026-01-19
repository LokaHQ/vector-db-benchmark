from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

ModelProvider = Literal["local", "bedrock"]


@dataclass
class ModelConfig:
    """Configuration for a specific embedding model."""

    name: str
    provider: ModelProvider
    dimensions: int
    normalize: bool = False


# Embedding model configurations
# Add your own models here by following these patterns:
#
# Local model (runs on your machine via sentence-transformers):
# "your-model-name": ModelConfig(
#     name="sentence-transformers/model-name",  # HuggingFace model ID
#     provider="local",
#     dimensions=768,  # Output embedding dimensions
#     normalize=False,  # Whether to normalize embeddings
# )
#
# Bedrock model (AWS managed):
# "your-model-name": ModelConfig(
#     name="bedrock-model-id",  # e.g., "amazon.titan-embed-text-v2:0"
#     provider="bedrock",
#     dimensions=1024,
#     normalize=True,
# )
MODEL_CONFIGS: dict[str, ModelConfig] = {
    # Example: Lightweight local model (384 dimensions)
    "minilm": ModelConfig(
        name="sentence-transformers/all-MiniLM-L6-v2",
        provider="local",
        dimensions=384,
    ),
    # Example: Higher quality local model (768 dimensions)
    "mpnet": ModelConfig(
        name="sentence-transformers/all-mpnet-base-v2",
        provider="local",
        dimensions=768,
    ),
    # Example: AWS Titan Embeddings v2 via Bedrock
    "titan-v2": ModelConfig(
        name="amazon.titan-embed-text-v2:0",
        provider="bedrock",
        dimensions=512,
        normalize=True,
    ),
    # Add more models here as needed
}

SupportedModels = Enum("SupportedModels", {key: key for key in MODEL_CONFIGS})


@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""

    chunking_type: Literal["character", "token"] = "character"
    chunk_size: int = 1000
    chunk_overlap: int = 200


@dataclass
class EmbeddingConfig:
    """Configuration for embedding providers."""

    model_name: str
    chunking_strategy: ChunkingConfig = field(default_factory=ChunkingConfig)
    max_concurrent_requests: int = 50
    provider: ModelProvider = "local"
    dimensions: int | None = None
    normalize: bool = True
