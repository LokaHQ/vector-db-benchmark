import time
from urllib.parse import urlparse

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from vector_db_benchmark.config import config as app_config
from vector_db_benchmark.utils.logging_config import log


def parse_s3_path(s3_path: str) -> tuple[str, str]:
    """Parses an S3 URI into bucket and key."""
    parsed_url = urlparse(s3_path)
    if parsed_url.scheme != "s3":
        raise ValueError(f"Invalid S3 URI: {s3_path}")
    return parsed_url.netloc, parsed_url.path.lstrip("/")


class AWSClient:
    """Centralized AWS client for S3 and Bedrock operations."""

    def __init__(self, region_name: str | None = None):
        """
        Initializes AWS clients with optimized connection pooling.

        Args:
            region_name: AWS region name. If not provided, uses the region from centralized config.
        """
        self.region_name = region_name or app_config.aws_region

        config = Config(
            max_pool_connections=50, retries={"max_attempts": 3, "mode": "adaptive"}
        )

        self.s3 = boto3.client("s3", region_name=self.region_name, config=config)
        self.bedrock = boto3.client(
            "bedrock-runtime", region_name=self.region_name, config=config
        )

    def read_s3_object_bytes(self, s3_path: str) -> bytes | None:
        """Reads a file's content directly from an S3 path into memory."""
        try:
            bucket, key = parse_s3_path(s3_path)
            response = self.s3.get_object(Bucket=bucket, Key=key)
            return response["Body"].read()
        except self.s3.exceptions.NoSuchKey:
            log.warning(f"File not found in S3: {s3_path}")
            return None
        except Exception as e:
            log.error(f"Failed to read file from S3: {s3_path}. Error: {e}")
            raise

    def write_s3_object_bytes(self, s3_path: str, content: bytes) -> None:
        """Writes raw bytes to a specific S3 object."""
        try:
            bucket, key = parse_s3_path(s3_path)
            self.s3.put_object(Bucket=bucket, Key=key, Body=content)
        except Exception as e:
            log.error(f"Failed to write to S3 path {s3_path}: {e}")
            raise

    def list_s3_prefix(self, bucket: str, prefix: str) -> list[dict]:
        """Lists all files in an S3 bucket with a given prefix."""
        try:
            paginator = self.s3.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
            s3_objects = []
            for page in pages:
                if "Contents" in page:
                    s3_objects.extend(page["Contents"])
            return s3_objects
        except Exception as e:
            log.error(
                f"Failed to list files from S3 bucket '{bucket}' with prefix '{prefix}'. Error: {e}"
            )
            return []

    def delete_objects_by_prefix(self, bucket: str, prefix: str) -> None:
        """Deletes all objects in an S3 bucket with a given prefix."""
        log.info(
            f"Deleting S3 objects from bucket '{bucket}' with prefix '{prefix}'..."
        )

        s3_objects = self.list_s3_prefix(bucket, prefix)

        if not s3_objects:
            log.info("No objects found to delete.")
            return

        objects_to_delete = [{"Key": obj["Key"]} for obj in s3_objects]

        for i in range(0, len(objects_to_delete), 1000):
            chunk = objects_to_delete[i : i + 1000]
            self.s3.delete_objects(
                Bucket=bucket, Delete={"Objects": chunk, "Quiet": True}
            )

        log.info(f"Successfully deleted {len(objects_to_delete)} object(s).")

    def bedrock_embed(self, model_id: str, body: str) -> dict:
        """Calls Bedrock embedding API with retry logic."""

        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                return self.bedrock.invoke_model(
                    modelId=model_id,
                    body=body,
                    accept="application/json",
                    contentType="application/json",
                )
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")

                if error_code in ["ThrottlingException", "TooManyRequestsException"]:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2**attempt)
                        log.warning(
                            f"Rate limited, retrying in {delay}s (attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(delay)
                        continue
                    log.error(f"Rate limit exceeded after {max_retries} attempts")
                    raise

                log.error(f"Bedrock API error: {error_code} - {e}")
                raise

            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    log.warning(f"Bedrock API error, retrying in {delay}s: {e}")
                    time.sleep(delay)
                    continue
                log.error(f"Bedrock API failed after {max_retries} attempts: {e}")
                raise
        return None

    def invoke_llm(self, model_id: str, body: str) -> dict:
        """Invokes an LLM model via Bedrock with retry logic."""

        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                # Try Converse API first (handles inference profiles automatically)
                if "anthropic" in model_id:
                    import json

                    body_dict = json.loads(body)
                    messages = body_dict.get("messages", [])
                    max_tokens = body_dict.get("max_tokens", 1024)

                    # Convert InvokeModel message format to Converse API format
                    converse_messages = []
                    for msg in messages:
                        if isinstance(msg["content"], str):
                            # Convert string content to list format expected by Converse API
                            converse_messages.append(
                                {
                                    "role": msg["role"],
                                    "content": [{"text": msg["content"]}],
                                }
                            )
                        else:
                            # Already in correct format
                            converse_messages.append(msg)

                    return self.bedrock.converse(
                        modelId=model_id,
                        messages=converse_messages,
                        inferenceConfig={"maxTokens": max_tokens},
                    )
                # Fallback to invoke_model for other models
                return self.bedrock.invoke_model(
                    modelId=model_id,
                    body=body,
                    accept="application/json",
                    contentType="application/json",
                )
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")

                if error_code in ["ThrottlingException", "TooManyRequestsException"]:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2**attempt)
                        log.warning(
                            f"Rate limited, retrying in {delay}s (attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(delay)
                        continue
                    log.error(f"Rate limit exceeded after {max_retries} attempts")
                    raise

                log.error(f"Bedrock API error: {error_code} - {e}")
                raise

            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    log.warning(f"Bedrock API error, retrying in {delay}s: {e}")
                    time.sleep(delay)
                    continue
                log.error(f"Bedrock API failed after {max_retries} attempts: {e}")
                raise
        return None
