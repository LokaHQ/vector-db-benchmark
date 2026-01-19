from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class CompletionResult:
    """Standardized output for a completion model call."""

    answer: str
    input_tokens: int
    output_tokens: int
    latency: float


class BaseCompletionProvider(ABC):
    """Abstract base class for a completion provider."""

    @abstractmethod
    def complete(self, query: str, context: list[str]) -> CompletionResult:
        """
        Generates an answer based on a query and a list of context strings.

        Args:
            query: The user's question.
            context: A list of text chunks retrieved from the vector database.

        Returns:
            A CompletionResult object with the generated answer and performance metrics.
        """
