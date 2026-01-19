from vector_db_benchmark.embedding.base import EmbeddingProvider
from vector_db_benchmark.embedding.bedrock_embedder import BedrockEmbedder
from vector_db_benchmark.embedding.config import MODEL_CONFIGS, EmbeddingConfig
from vector_db_benchmark.services.aws import AWSClient

try:
    from vector_db_benchmark.embedding.local_embedder import LocalEmbedder
except ImportError:
    LocalEmbedder = None  # type: ignore

PROVIDER_CLASSES = {
    "local": LocalEmbedder,
    "bedrock": BedrockEmbedder,
}


def create_embedder(model_name: str) -> EmbeddingProvider:
    """
    Factory function that returns a fully configured embedding provider instance.

    Args:
        model_name: The full name of the model to use (e.g., "sentence-transformers/all-MiniLM-L6-v2").

    Returns:
        An instance of an EmbeddingProvider.

    Raises:
        ValueError: If the model name is unknown.
    """
    model_config = next(
        (mc for mc in MODEL_CONFIGS.values() if mc.name == model_name), None
    )

    if not model_config:
        raise ValueError(f"Unknown embedding model name: {model_name}")

    config = EmbeddingConfig(
        model_name=model_config.name,
        provider=model_config.provider,
        dimensions=model_config.dimensions,
        normalize=model_config.normalize,
    )

    provider_class = PROVIDER_CLASSES.get(model_config.provider)
    if not provider_class:
        raise ValueError(
            f"No provider class found for provider: {model_config.provider}"
        )

    if provider_class is None:
        raise ImportError(
            f"Provider '{model_config.provider}' is not available. "
            "Please install the required dependencies with: "
            "`uv sync --extra local_models`"
        )

    if model_config.provider == "bedrock":
        aws_client = AWSClient()
        return provider_class(config=config, aws_client=aws_client)

    return provider_class(config=config)
