from vector_db_benchmark.completion.base import BaseCompletionProvider
from vector_db_benchmark.completion.bedrock import BedrockCompletionProvider
from vector_db_benchmark.completion.config import COMPLETION_MODEL_CONFIGS
from vector_db_benchmark.services.aws import AWSClient

PROVIDER_CLASSES = {
    "bedrock": BedrockCompletionProvider,
}


def create_completion_provider(model_name: str) -> BaseCompletionProvider:
    """
    Factory function that returns a fully configured completion provider instance.

    Args:
        model_name: The friendly name of the model to use (e.g., "claude-3-sonnet").

    Returns:
        An instance of a BaseCompletionProvider.

    Raises:
        ValueError: If the model name is unknown or the provider is not supported.
    """
    if model_name not in COMPLETION_MODEL_CONFIGS:
        raise ValueError(f"Unknown completion model name: {model_name}")

    config = COMPLETION_MODEL_CONFIGS[model_name]
    provider_class = PROVIDER_CLASSES.get(config.provider)

    if not provider_class:
        raise ValueError(f"No provider class found for provider: {config.provider}")

    if config.provider == "bedrock":
        aws_client = AWSClient()
        return provider_class(config=config, aws_client=aws_client)

    return provider_class(config=config)
