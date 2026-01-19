import contextlib
import shutil
from typing import Any

import lancedb
import pandas as pd

from vector_db_benchmark.utils.logging_config import log

from .base import VectorDatabase
from .config import DatabaseConfig


class LanceVectorDB(VectorDatabase):
    """LanceDB vector database."""

    def __init__(self, config: DatabaseConfig):
        super().__init__(config)

        self.uri = self._setup_storage_path(config.path)

        self.client = lancedb.connect(self.uri)
        self.table = None  # Will be set when collection is created

    @property
    def name(self) -> str:
        return "lance"

    def save(self, path: str) -> None:
        """Saves the database to a local path."""
        shutil.copytree(self.uri, path, dirs_exist_ok=True)
        self._save_metadata(path)

    @classmethod
    def load(cls, path: str, config: DatabaseConfig) -> "LanceVectorDB":
        """Loads the database from a local path."""
        metadata = cls._load_metadata(path)

        config.path = path
        db = cls(config)

        db.collection_name = metadata.get("collection_name")
        if db.collection_name:
            db.table = db.client.open_table(db.collection_name)
            log.info(f"Loaded LanceDB table: {db.collection_name}")

        return db

    def create_collection(
        self, collection_name: str, embedding_dim: int  # noqa: ARG002
    ) -> None:
        """Creates a collection in LanceDB."""
        self.collection_name = collection_name

        with contextlib.suppress(Exception):
            self.client.drop_table(collection_name, ignore_missing=True)

        log.info(f"Prepared LanceDB collection: {collection_name}")

    def insert_chunks(self, chunks: list[dict[str, Any]]):
        """Inserts chunks into the table."""
        if not chunks:
            return

        rows = [
            {
                "id": self._generate_chunk_id(chunk, i),
                "vector": chunk["embedding"],
                "text": chunk["text"],
            }
            for i, chunk in enumerate(chunks)
        ]

        df = pd.DataFrame(rows)

        if self.table is None:
            self.table = self.client.create_table(self.collection_name, data=df)
            log.info(f"Created LanceDB table: {self.collection_name}")
        else:
            self.table.add(df)

        log.info(f"Inserted {len(chunks)} chunks into LanceDB")

    def search(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[tuple[float, str, dict]]:
        """Searches for similar vectors."""
        if self.table is None:
            return []

        results_df = self.table.search(query_embedding).limit(top_k).to_pandas()

        if results_df.empty:
            return []

        results = []
        for _, row in results_df.iterrows():
            distance = float(row.get("_distance", 0.0))
            similarity_score = 1.0 - distance
            text = row.get("text", "")
            metadata = {
                k: v
                for k, v in row.items()
                if k not in ["vector", "text", "_distance", "distance"]
            }

            results.append((similarity_score, text, metadata))

        return results
