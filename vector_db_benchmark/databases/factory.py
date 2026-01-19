from vector_db_benchmark.databases.base import VectorDatabase
from vector_db_benchmark.databases.chroma_db import ChromaVectorDB
from vector_db_benchmark.databases.lance_db import LanceVectorDB
from vector_db_benchmark.databases.milvus_lite_db import MilvusLiteVectorDB
from vector_db_benchmark.databases.qdrant_db import QdrantVectorDB
from vector_db_benchmark.databases.s3_vectors_db import S3VectorsDB

PROVIDER_CLASSES = {
    "chroma": ChromaVectorDB,
    "qdrant": QdrantVectorDB,
    "milvus_lite": MilvusLiteVectorDB,
    "lance": LanceVectorDB,
    "s3_vectors": S3VectorsDB,
}


def get_database_provider(provider_name: str) -> type[VectorDatabase]:
    """Returns the database provider class for the given provider name."""
    if provider_name not in PROVIDER_CLASSES:
        raise ValueError(f"Unknown database provider: {provider_name}")
    return PROVIDER_CLASSES[provider_name]
