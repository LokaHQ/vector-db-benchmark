from dataclasses import dataclass, field
from typing import Any

from vector_db_benchmark.config import config as app_config


@dataclass
class DatabaseConfig:
    """
    A dataclass to hold configuration for a vector database.

    This provides a consistent structure for future additions that might
    need connection URLs, API keys, etc., without changing method signatures.
    """

    path: str | None = None
    host: str = "localhost"
    port: int = 6333
    vector_bucket: str | None = None
    region: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Auto-populate S3 Vectors config from centralized config if not explicitly set."""
        if self.vector_bucket is None:
            self.vector_bucket = app_config.s3_vector_bucket
        if self.region is None:
            self.region = app_config.s3_vector_region
