import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """
    Central configuration for the benchmark tool.

    All settings are loaded from environment variables with local-first defaults.
    S3/AWS settings are only required when using S3 storage mode or S3-based databases.
    """

    # Local defaults
    default_docs_dir: str = "./data/documents"
    default_results_dir: str = "./data/results"
    default_storage_mode: str = "local"

    # S3 Configuration (optional - only needed for S3 mode)
    s3_bucket: Optional[str] = None
    s3_vector_bucket: Optional[str] = None

    # AWS Configuration
    aws_region: str = "us-east-1"
    s3_vector_region: str = "us-east-1"

    def __post_init__(self):
        """Load configuration from environment variables."""
        # S3 bucket for general benchmark storage (results, persisted databases)
        self.s3_bucket = os.getenv("S3_BUCKET", self.s3_bucket)

        # S3 bucket for S3 Vectors database specifically
        self.s3_vector_bucket = os.getenv("S3_VECTOR_BUCKET", self.s3_vector_bucket)

        # AWS regions
        self.aws_region = os.getenv("AWS_REGION", self.aws_region)
        self.s3_vector_region = os.getenv("S3_VECTOR_REGION", self.s3_vector_region)

    def get_default_docs_dir(self) -> str:
        """
        Get the default documents directory.
        Returns local path by default, or S3 path if S3_BUCKET is configured.
        """
        return self.default_docs_dir

    def get_default_results_dir(self) -> str:
        """
        Get the default results directory.
        Returns local path by default, or S3 path if S3_BUCKET is configured.
        """
        return self.default_results_dir

    def get_s3_docs_dir(self) -> str:
        """
        Get the S3 documents directory path.
        Raises ValueError if S3_BUCKET is not configured.
        """
        if not self.s3_bucket:
            raise ValueError(
                "S3_BUCKET environment variable must be set to use S3 paths. "
                "Please set it in your .env file or environment."
            )
        return f"s3://{self.s3_bucket}/documents"

    def get_s3_results_dir(self) -> str:
        """
        Get the S3 results directory path.
        Raises ValueError if S3_BUCKET is not configured.
        """
        if not self.s3_bucket:
            raise ValueError(
                "S3_BUCKET environment variable must be set to use S3 paths. "
                "Please set it in your .env file or environment."
            )
        return f"s3://{self.s3_bucket}/results"

    def validate_s3_config(self) -> None:
        """
        Validate that required S3 configuration is present.
        Should be called when storage_mode="s3" or when using S3 Vectors database.
        """
        if not self.s3_bucket:
            raise ValueError(
                "S3_BUCKET environment variable is required when using S3 storage mode. "
                "Please set it in your .env file or environment."
            )

    def validate_s3_vectors_config(self) -> None:
        """
        Validate that S3 Vectors database configuration is present.
        Should be called when using the S3 Vectors database provider.
        """
        if not self.s3_vector_bucket:
            raise ValueError(
                "S3_VECTOR_BUCKET environment variable is required when using S3 Vectors database. "
                "Please set it in your .env file or environment."
            )


# Global configuration instance
config = Config()
