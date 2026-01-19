import uuid
from typing import Any

import boto3

from vector_db_benchmark.utils.logging_config import log

from .base import VectorDatabase
from .config import DatabaseConfig


class S3VectorsDB(VectorDatabase):
    """
    AWS S3 Vectors database - native cloud vector storage.

    Unlike traditional vector DBs, S3 Vectors is already persistent in S3.
    No save/load operations needed - vectors are written/queried directly.
    """

    def __init__(self, config: DatabaseConfig):
        super().__init__(config)

        if not config.vector_bucket:
            raise ValueError("S3 Vectors requires vector_bucket in config")

        self.vector_bucket = config.vector_bucket
        self.region = config.region

        # Initialize S3 Vectors client
        self.client = boto3.client("s3vectors", region_name=self.region)

        log.info(f"Initialized S3 Vectors client for bucket: {self.vector_bucket}")

    @property
    def name(self) -> str:
        return "s3_vectors"

    def supports_direct_serialization(self) -> bool:
        """S3 Vectors doesn't need serialization - it's already in S3."""
        return False

    def requires_persistence(self) -> bool:
        """
        S3 Vectors requires persistence of metadata (not data).
        We need to save/load the index name to reconnect.
        """
        return True

    def save(self, path: str) -> None:
        """
        S3 Vectors doesn't need to save data (already in S3),
        but we save metadata so we can reconnect to the same index.
        """
        log.debug(
            f"S3 Vectors save() - saving metadata for index: {self.collection_name}"
        )
        # Use base class method which handles local file writing
        self._save_metadata(path)

    @classmethod
    def load(cls, path: str, config: DatabaseConfig) -> "S3VectorsDB":
        """
        S3 Vectors reconnects to existing index by loading metadata.
        """
        log.debug("S3 Vectors load() - loading metadata to reconnect")
        # Use base class method which handles local file reading
        metadata = cls._load_metadata(path)

        db = cls(config)
        db.collection_name = metadata.get("collection_name")
        log.info(f"Reconnected to S3 Vectors index: {db.collection_name}")
        return db

    def create_collection(self, collection_name: str, embedding_dim: int) -> None:
        """Creates an S3 vector index (collection)."""
        # S3 Vectors index names must follow S3 bucket naming rules:
        # - lowercase letters, numbers, hyphens (no underscores)
        # - 3-63 characters
        # - start with letter or number
        sanitized_name = collection_name.replace("_", "-").lower()
        # Truncate to 63 characters max
        if len(sanitized_name) > 63:
            sanitized_name = sanitized_name[:63]
        # Ensure it doesn't end with a hyphen after truncation
        sanitized_name = sanitized_name.rstrip("-")
        self.collection_name = sanitized_name

        log.info(
            f"DEBUG: Creating S3 Vectors index with name: '{sanitized_name}' (length: {len(sanitized_name)})"
        )
        log.info(f"DEBUG: Using vector bucket: '{self.vector_bucket}'")
        log.info(f"DEBUG: Using region: '{self.region}'")

        # Check if index already exists and delete if needed
        try:
            existing_indexes = self.client.list_indexes(
                vectorBucketName=self.vector_bucket
            )
            index_names = [
                idx["indexName"] for idx in existing_indexes.get("indexes", [])
            ]

            if sanitized_name in index_names:
                log.debug(f"Deleting existing S3 vector index: {sanitized_name}")
                self.client.delete_index(
                    vectorBucketName=self.vector_bucket, indexName=sanitized_name
                )
                log.debug(f"Deleted existing index: {sanitized_name}")
        except Exception as e:
            log.debug(f"Index check/deletion skipped: {e}")

        # Create new index with specified dimensions
        # Using cosine distance as default (consistent with Chroma/Qdrant defaults)
        self.client.create_index(
            vectorBucketName=self.vector_bucket,
            indexName=sanitized_name,
            dataType="float32",
            dimension=embedding_dim,
            distanceMetric="cosine",
            metadataConfiguration={"nonFilterableMetadataKeys": ["text"]},
        )

        log.info(
            f"Created S3 vector index: {collection_name} "
            f"(bucket: {self.vector_bucket}, dim: {embedding_dim})"
        )

    def insert_chunks(self, chunks: list[dict[str, Any]]) -> int:
        """Inserts chunks directly into S3 vector index."""
        if not self.collection_name:
            raise ValueError("Collection not created. Call create_collection first.")

        # S3 Vectors put_vectors supports up to 500 vectors per batch
        batch_size = 500
        total_inserted = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            vectors_to_insert = []

            for chunk in batch:
                # Generate unique key for this vector
                vector_key = str(uuid.uuid4())

                # Extract metadata (excluding embedding and text - text goes in metadata)
                metadata = self._extract_metadata(chunk, exclude_fields=["embedding"])

                vector_data = {
                    "key": vector_key,
                    "data": {"float32": chunk["embedding"]},
                    "metadata": metadata,
                }

                vectors_to_insert.append(vector_data)

            # Insert batch to S3 Vectors
            self.client.put_vectors(
                vectorBucketName=self.vector_bucket,
                indexName=self.collection_name,
                vectors=vectors_to_insert,
            )

            total_inserted += len(batch)
            log.debug(f"Inserted batch of {len(batch)} vectors to S3 Vectors")

        log.info(
            f"Inserted {total_inserted} chunks into S3 Vectors index: {self.collection_name}"
        )
        return total_inserted

    def search(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[tuple[float, str, dict]]:
        """Searches for similar vectors directly in S3 Vectors."""
        if not self.collection_name:
            raise ValueError("Collection not created. Call create_collection first.")

        # Query S3 Vectors with metadata and distance
        response = self.client.query_vectors(
            vectorBucketName=self.vector_bucket,
            indexName=self.collection_name,
            queryVector={"float32": query_embedding},
            topK=top_k,
            returnMetadata=True,
            returnDistance=True,
        )

        results = []
        for vector_result in response.get("vectors", []):
            # S3 Vectors returns distance (lower is better for cosine)
            # Convert to similarity score (higher is better)
            distance = vector_result.get("distance", 1.0)
            similarity_score = 1.0 - distance

            # Extract text and metadata
            metadata = vector_result.get("metadata", {})
            text = metadata.pop("text", "")

            results.append((similarity_score, text, metadata))

        log.debug(f"S3 Vectors search returned {len(results)} results")
        return results
