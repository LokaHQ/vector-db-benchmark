from dataclasses import dataclass
from typing import Literal

CompletionProvider = Literal["bedrock"]


@dataclass
class CompletionModelConfig:
    """Configuration for a specific completion model."""

    name: str
    model_id: str
    provider: CompletionProvider


# Completion model configurations
# Add your own models here by following this pattern:
# "your-model-name": CompletionModelConfig(
#     name="display-name",
#     model_id="bedrock-model-id-or-arn",  # e.g., "anthropic.claude-3-sonnet-20240229-v1:0"
#     provider="bedrock",
# )
COMPLETION_MODEL_CONFIGS: dict[str, CompletionModelConfig] = {
    # Example: Claude 3.5 Sonnet via Bedrock cross-region inference profile
    "claude-3-sonnet": CompletionModelConfig(
        name="claude-3-sonnet",
        model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",  # Cross-region inference profile
        provider="bedrock",
    ),
    # Example: Llama 3 8B via Bedrock
    "llama3-8b": CompletionModelConfig(
        name="llama3-8b",
        model_id="meta.llama3-8b-instruct-v1:0",
        provider="bedrock",
    ),
    # Add more models here as needed
}
