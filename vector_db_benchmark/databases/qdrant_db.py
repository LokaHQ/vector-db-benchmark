import pickle
import shutil
import uuid
from typing import Any

from qdrant_client import QdrantClient, models

from vector_db_benchmark.utils.logging_config import log

from .base import VectorDatabase
from .config import DatabaseConfig


class QdrantVectorDB(VectorDatabase):
    """Qdrant vector database."""

    def __init__(self, config: DatabaseConfig):
        super().__init__(config)

        if config.path:
            self.path = self._setup_storage_path(config.path)
            self.client = QdrantClient(path=self.path)
        else:
            self.path = None
            self.client = QdrantClient(location=":memory:")
        self.collection_name = None

    @property
    def name(self) -> str:
        return "qdrant"

    def supports_direct_serialization(self) -> bool:
        return self.path is None

    def serialize_to_bytes(self) -> bytes:
        if not self.supports_direct_serialization():
            raise ValueError("Direct serialization only supported for in-memory mode")

        points_data = []
        collection_info = None

        if self.collection_name:
            collection_info = self.client.get_collection(self.collection_name)
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,
                with_vectors=True,
                with_payload=True,
            )
            points_data = [(point.id, point.vector, point.payload) for point in points]

        serialized_data = {
            "collection_name": self.collection_name,
            "collection_info": collection_info,
            "points": points_data,
            "metadata": {"db_type": self.name, "version": "1.0"},
        }
        return pickle.dumps(serialized_data)

    def deserialize_from_bytes(self, data_bytes: bytes) -> None:
        serialized_data = pickle.loads(data_bytes)  # noqa: S301
        self.collection_name = serialized_data["collection_name"]

        if self.collection_name and serialized_data.get("collection_info"):
            collection_info = serialized_data["collection_info"]
            vector_size = collection_info.config.params.vectors.size

            # Delete existing collection first to ensure clean state
            self._safe_delete_collection(
                self.collection_name, lambda name: self.client.delete_collection(name)
            )

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size, distance=models.Distance.COSINE
                ),
            )

            if serialized_data["points"]:
                points = [
                    models.PointStruct(id=point_id, vector=vector, payload=payload)
                    for point_id, vector, payload in serialized_data["points"]
                ]
                self.client.upsert(collection_name=self.collection_name, points=points)

    def save(self, path: str) -> None:
        """Saves the database to a local path."""
        if self.path and path:
            shutil.copytree(self.path, path, dirs_exist_ok=True)
        self._save_metadata(path)

    @classmethod
    def load(cls, path: str, config: DatabaseConfig) -> "QdrantVectorDB":
        """Loads the database from a local path."""
        metadata = cls._load_metadata(path)
        config.path = path
        db = cls(config)
        db.collection_name = metadata.get("collection_name")
        return db

    def create_collection(self, collection_name: str, embedding_dim: int) -> None:
        """Creates or recreates a collection."""
        self.collection_name = collection_name

        self._safe_delete_collection(
            collection_name,
            lambda name: self.client.delete_collection(collection_name=name),
        )

        if embedding_dim is None:
            raise ValueError("Qdrant requires vector dimension to be specified")

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=embedding_dim, distance=models.Distance.COSINE
            ),
        )
        log.info(
            f"Created Qdrant collection: {collection_name} with dimension {embedding_dim}"
        )

    def insert_chunks(self, chunks: list[dict[str, Any]]) -> int:
        """Inserts chunks into the collection."""
        if not self.collection_name:
            raise ValueError("Collection not created. Call create_collection first.")

        points = []

        for _, chunk in enumerate(chunks):
            point_id = str(uuid.uuid4())
            payload = self._extract_metadata(chunk, exclude_fields=["embedding"])

            point = models.PointStruct(
                id=point_id, vector=chunk["embedding"], payload=payload
            )
            points.append(point)

        self.client.upsert(collection_name=self.collection_name, points=points)

        log.info(f"Inserted {len(chunks)} chunks into Qdrant")
        return len(chunks)

    def search(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[tuple[float, str, dict]]:
        """Searches for similar vectors."""
        if not self.collection_name:
            raise ValueError("Collection not created. Call create_collection first.")

        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True,
        )

        results = []
        for result in search_results:
            score = result.score
            text = result.payload.get("text", "")
            metadata = {k: v for k, v in result.payload.items() if k != "text"}

            results.append((score, text, metadata))

        return results
