"""Bedrock completion provider."""

import json
import time

from vector_db_benchmark.completion.base import BaseCompletionProvider, CompletionResult
from vector_db_benchmark.completion.config import CompletionModelConfig
from vector_db_benchmark.services.aws import AWSClient
from vector_db_benchmark.utils.logging_config import log


class BedrockCompletionProvider(BaseCompletionProvider):
    """Bedrock completion provider."""

    def __init__(
        self, config: CompletionModelConfig, aws_client: AWSClient | None = None
    ):
        """Initialize the Bedrock completion provider."""
        self.config = config
        self.aws_client = aws_client or AWSClient()

    def _build_prompt(self, query: str, context: list[str]) -> str:
        """Builds a context-aware prompt for the LLM."""
        context_str = "\n".join(context)
        return f"""
        Human: You are an expert question-answering AI assistant.
        Use the following context to answer the user's question.
        If the answer is not present in the context, say that you don't know.

        Context:
        {context_str}

        Question: {query}

        Assistant:
        """

    def _invoke_claude(self, prompt: str) -> CompletionResult:
        """Invokes a Claude model on Bedrock."""
        body = json.dumps(
            {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024,
                "anthropic_version": "bedrock-2023-05-31",
            }
        )

        start_time = time.time()
        response = self.aws_client.invoke_llm(model_id=self.config.model_id, body=body)
        latency = time.time() - start_time

        # Handle both Converse API and InvokeModel API responses
        if "output" in response:
            # Converse API response
            answer = response["output"]["message"]["content"][0]["text"]
            input_tokens = response["usage"]["inputTokens"]
            output_tokens = response["usage"]["outputTokens"]
        else:
            # InvokeModel API response
            response_body = json.loads(response["body"].read())
            answer = response_body["content"][0]["text"]
            input_tokens = response_body["usage"]["input_tokens"]
            output_tokens = response_body["usage"]["output_tokens"]

        return CompletionResult(
            answer=answer,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency=latency,
        )

    def _invoke_llama(self, prompt: str) -> CompletionResult:
        """Invokes a Llama model on Bedrock."""
        body = json.dumps(
            {
                "prompt": prompt,
                "max_gen_len": 1024,
                "temperature": 0.1,
            }
        )

        start_time = time.time()
        response = self.aws_client.invoke_llm(model_id=self.config.model_id, body=body)
        latency = time.time() - start_time

        response_body = json.loads(response["body"].read())
        answer = response_body["generation"]
        input_tokens = response_body["prompt_token_count"]
        output_tokens = response_body["generation_token_count"]

        return CompletionResult(
            answer=answer,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency=latency,
        )

    def complete(self, query: str, context: list[str]) -> CompletionResult:
        """Generates an answer using the configured Bedrock model."""
        prompt = self._build_prompt(query, context)
        log.info(f"Invoking {self.config.model_id} with prompt...")

        if "anthropic" in self.config.model_id:
            return self._invoke_claude(prompt)
        if "meta" in self.config.model_id:
            return self._invoke_llama(prompt)

        raise NotImplementedError(
            f"Model provider for {self.config.model_id} not implemented."
        )
