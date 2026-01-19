import os
import shutil
from typing import Any

from pymilvus import (
    MilvusClient,
)

from vector_db_benchmark.utils.logging_config import log

from .base import VectorDatabase
from .config import DatabaseConfig


class MilvusLiteVectorDB(VectorDatabase):
    """Milvus Lite vector database."""

    def __init__(self, config: DatabaseConfig):
        super().__init__(config)

        # Create database file path
        if config.path:
            self.db_file = config.path
        else:
            temp_dir = self._setup_storage_path(None)
            self.db_file = os.path.join(temp_dir, "milvus_lite.db")

        # Initialize Milvus Lite client
        self.client = MilvusClient(uri=self.db_file)

    @property
    def name(self) -> str:
        return "milvus_lite"

    def save(self, path: str) -> None:
        """Saves the database to a local path."""
        self.client.close()
        self.client = None
        shutil.copy(self.db_file, os.path.join(path, "milvus_lite.db"))
        self._save_metadata(path)

    @classmethod
    def load(cls, path: str, config: DatabaseConfig) -> "MilvusLiteVectorDB":
        """Loads the database from a local path."""
        metadata = cls._load_metadata(path)
        db_file_path = os.path.join(path, "milvus_lite.db")

        config.path = db_file_path
        db = cls(config)
        db.collection_name = metadata.get("collection_name")

        return db

    def create_collection(self, collection_name: str, embedding_dim: int):
        """Creates a collection in Milvus."""
        self.collection_name = collection_name
        self._safe_delete_collection(
            collection_name,
            lambda name: (
                self.client.drop_collection(name)
                if self.client.has_collection(name)
                else None
            ),
        )
        self.client.create_collection(
            collection_name=collection_name,
            dimension=embedding_dim,
            primary_field_name="id",
            id_type="str",
            max_length=128,
            vector_field_name="vector",
            metric_type="COSINE",
        )
        log.info(
            f"Created Milvus Lite collection: {collection_name} (dim={embedding_dim})"
        )

    def insert_chunks(self, chunks: list[dict[str, Any]]):
        """Inserts chunks into the collection."""
        if not self.collection_name:
            raise ValueError("Collection not created. Call create_collection first.")
        data = []
        for i, chunk in enumerate(chunks):
            chunk_id = self._generate_chunk_id(chunk, i)
            row = {"id": chunk_id, "vector": chunk["embedding"], "text": chunk["text"]}
            metadata = self._extract_metadata(
                chunk, exclude_fields=["embedding", "text", "chunk_id", "document_name"]
            )
            row.update(metadata)
            data.append(row)
        self.client.insert(collection_name=self.collection_name, data=data)
        log.info(f"Inserted {len(chunks)} chunks into Milvus Lite")

    def search(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[tuple[float, str, dict]]:
        """Searches for similar vectors."""
        if not self.collection_name:
            raise ValueError("Collection not created. Call create_collection first.")

        res = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=top_k,
            output_fields=["*"],
        )

        results = []
        if res and res[0]:
            for hit in res[0]:
                similarity_score = 1.0 - hit.get("distance", 1.0)
                entity = hit.get("entity", {})
                text = entity.get("text", "")
                metadata = {
                    k: v for k, v in entity.items() if k not in ["text", "vector"]
                }
                metadata["id"] = hit.get("id", "")

                results.append((similarity_score, text, metadata))

        return results
