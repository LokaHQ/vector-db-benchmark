import json
import os
import tempfile
from abc import ABC, abstractmethod
from typing import Any

from vector_db_benchmark.utils.logging_config import log

from .config import DatabaseConfig


class VectorDatabase(ABC):
    """Abstract base class for a vector database."""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.client = None
        self.collection = None
        self.collection_name = None
        self._temp_dir = None

    def __enter__(self):
        """Enter the runtime context related to this object."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the runtime context, cleaning up resources."""
        if self._temp_dir:
            self._temp_dir.cleanup()
            log.debug(f"Cleaned up temporary directory: {self._temp_dir.name}")

    def _setup_storage_path(self, config_path: str | None) -> str:
        """Sets up storage path, creating a temporary directory if needed."""
        if config_path:
            self._temp_dir = None
            return config_path
        self._temp_dir = tempfile.TemporaryDirectory()
        return self._temp_dir.name

    def _generate_chunk_id(self, chunk: dict[str, Any], index: int) -> str:
        """Generates a consistent chunk ID from chunk data."""
        document_name = chunk.get("document_name", "doc")
        chunk_id = chunk.get("chunk_id", index)
        return f"{document_name}_{chunk_id}"

    def _extract_metadata(
        self, chunk: dict[str, Any], exclude_fields: list[str] = None
    ) -> dict[str, Any]:
        """Extracts metadata from chunk, excluding specified fields."""
        if exclude_fields is None:
            exclude_fields = ["embedding", "text"]

        return {k: v for k, v in chunk.items() if k not in exclude_fields}

    def _safe_delete_collection(self, collection_name: str, delete_func) -> None:
        """Safely deletes a collection, logging errors without failing."""
        try:
            delete_func(collection_name)
            log.debug(f"Successfully deleted collection: {collection_name}")
        except Exception as e:
            log.debug(
                f"Collection deletion failed (expected if collection does not exist): {e}"
            )

    def _save_metadata(self, path: str) -> None:
        """Saves essential metadata to a JSON file."""
        metadata = {
            "collection_name": self.collection_name,
            "name": self.name,
        }
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f)

    @staticmethod
    def _load_metadata(path: str) -> dict:
        """Loads metadata from a JSON file."""
        with open(os.path.join(path, "metadata.json")) as f:
            return json.load(f)

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the vector database."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Saves the database's state to the given local filesystem path."""

    @classmethod
    @abstractmethod
    def load(cls, path: str, config: DatabaseConfig) -> "VectorDatabase":
        """Loads the database from a local directory."""

    @abstractmethod
    def create_collection(self, collection_name: str, embedding_dim: int) -> None:
        """Creates a collection for storing vectors."""

    @abstractmethod
    def insert_chunks(self, chunks: list[dict]) -> int:
        """Inserts text chunks with embeddings into the database."""

    @abstractmethod
    def search(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[tuple[float, str, dict]]:
        """Searches for the top_k most similar vectors to the query embedding."""

    def supports_direct_serialization(self) -> bool:
        return False

    def serialize_to_bytes(self) -> bytes:
        raise NotImplementedError

    def deserialize_from_bytes(self, data_bytes: bytes) -> None:
        raise NotImplementedError

    def requires_persistence(self) -> bool:
        """
        Returns True if this database needs save/load operations.
        Returns False for cloud-native DBs like S3 Vectors that are already persistent.
        """
        return True
