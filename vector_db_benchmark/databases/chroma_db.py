import pickle
from typing import Any

import chromadb

from vector_db_benchmark.utils.logging_config import log

from .base import VectorDatabase
from .config import DatabaseConfig


class ChromaVectorDB(VectorDatabase):
    """ChromaDB vector database."""

    def __init__(self, config: DatabaseConfig):
        super().__init__(config)

        if config.path:
            self.db_path = self._setup_storage_path(config.path)
            self.client = chromadb.PersistentClient(path=self.db_path)
        else:
            self.db_path = None
            self.client = chromadb.Client(
                settings=chromadb.config.Settings(
                    anonymized_telemetry=False, allow_reset=True
                )
            )

    @property
    def name(self) -> str:
        return "chroma"

    def supports_direct_serialization(self) -> bool:
        return self.db_path is None

    def serialize_to_bytes(self) -> bytes:
        if not self.supports_direct_serialization():
            raise ValueError("Direct serialization only supported for in-memory mode")

        data = (
            self.collection.get(include=["embeddings", "documents", "metadatas"])
            if self.collection
            else None
        )
        serialized_data = {
            "collection_name": self.collection_name,
            "data": data,
            "metadata": {"db_type": self.name, "version": "1.0"},
        }
        return pickle.dumps(serialized_data)

    def deserialize_from_bytes(self, data_bytes: bytes) -> None:
        serialized_data = pickle.loads(data_bytes)
        self.collection_name = serialized_data["collection_name"]

        if self.collection_name:
            self._safe_delete_collection(
                self.collection_name,
                lambda name: self.client.delete_collection(name=name),
            )
            self.collection = self.client.create_collection(name=self.collection_name)
            data = serialized_data["data"]

            if data and data["ids"]:
                self.collection.add(
                    ids=data["ids"],
                    embeddings=data["embeddings"],
                    documents=data["documents"],
                    metadatas=data["metadatas"],
                )

    def save(self, path: str) -> None:
        """Saves the database to a local path."""
        log.info(f"Saving ChromaDB collection '{self.collection_name}' to path: {path}")
        persistent_client = chromadb.PersistentClient(path=path)

        if self.collection:
            data = self.collection.get(include=["embeddings", "documents", "metadatas"])

            persistent_collection = persistent_client.get_or_create_collection(
                name=self.collection_name
            )

            if data and data["ids"]:
                persistent_collection.add(
                    ids=data["ids"],
                    embeddings=data["embeddings"],
                    documents=data["documents"],
                    metadatas=data["metadatas"],
                )

        self._save_metadata(path)

    @classmethod
    def load(cls, path: str, config: DatabaseConfig) -> "ChromaVectorDB":
        """Loads the database from a local path."""
        metadata = cls._load_metadata(path)
        config.path = path

        db = cls(config)
        db.collection_name = metadata.get("collection_name")

        if db.collection_name:
            db.collection = db.client.get_collection(name=db.collection_name)

        return db

    def create_collection(
        self, collection_name: str, embedding_dim: int  # noqa: ARG002
    ) -> None:
        """Creates a collection."""
        self.collection_name = collection_name

        self._safe_delete_collection(
            collection_name, lambda name: self.client.delete_collection(name=name)
        )

        self.collection = self.client.create_collection(
            name=collection_name,
        )
        log.info(f"Created ChromaDB collection: {collection_name}")

    def insert_chunks(self, chunks: list[dict[str, Any]]) -> int:
        """Inserts chunks into the collection."""
        if not self.collection:
            raise ValueError("Collection not created. Call create_collection first.")

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = self._generate_chunk_id(chunk, i)
            ids.append(chunk_id)
            embeddings.append(chunk["embedding"])
            documents.append(chunk["text"])
            metadata = self._extract_metadata(chunk)
            metadatas.append(metadata)

        self.collection.add(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

        log.info(f"Inserted {len(chunks)} chunks into ChromaDB")
        return len(chunks)

    def search(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[tuple[float, str, dict]]:
        """Searches for similar vectors."""
        if not self.collection:
            raise ValueError("Collection not created. Call create_collection first.")

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        if results["distances"] and results["distances"][0]:
            for distance, document, metadata in zip(
                results["distances"][0],
                results["documents"][0],
                results["metadatas"][0],
                strict=True,
            ):
                similarity_score = 1.0 - distance
                search_results.append((similarity_score, document, metadata))

        return search_results
