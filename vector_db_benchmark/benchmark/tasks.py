import os
import tempfile
from itertools import product
from typing import Any

from vector_db_benchmark.config import config as app_config
from vector_db_benchmark.databases.base import VectorDatabase
from vector_db_benchmark.embedding.base import EmbeddingProvider
from vector_db_benchmark.utils.document_manager import DocumentInfo
from vector_db_benchmark.utils.logging_config import log
from vector_db_benchmark.utils.query_generator import QueryGenerator
from vector_db_benchmark.utils.text_extractor import PDFProcessor

from ..completion.config import COMPLETION_MODEL_CONFIGS
from ..embedding.config import MODEL_CONFIGS
from ..utils.document_manager import DocumentManager
from .schemas import BenchmarkJob


def _get_documents(args) -> list[DocumentInfo]:
    """Resolves document selection from command-line arguments."""
    try:
        doc_manager = DocumentManager(args.docs_dir)
    except FileNotFoundError as e:
        log.error(f"Failed to initialize DocumentManager: {e}")
        return []

    if args.docs == "sample":
        return doc_manager.get_sample_documents(per_category=1)
    if args.docs == "all":
        return doc_manager.get_all_documents()
    return doc_manager.get_documents_by_category(args.docs)


def _get_db_providers(args) -> list[str]:
    """Resolves database provider selection from command-line arguments."""
    return (
        ["chroma", "qdrant", "milvus_lite", "lance", "s3_vectors"]
        if "all" in args.dbs
        else args.dbs
    )


def _get_model_names(args) -> list[str]:
    """Resolves embedding model selection from command-line arguments."""
    model_choices = set(args.models)
    model_names = set()

    if "all" in model_choices:
        model_names.update(config.name for config in MODEL_CONFIGS.values())
    if "local" in model_choices:
        model_names.update(
            config.name
            for config in MODEL_CONFIGS.values()
            if config.provider == "local"
        )
    if "bedrock" in model_choices:
        model_names.update(
            config.name
            for config in MODEL_CONFIGS.values()
            if config.provider == "bedrock"
        )
    for choice in model_choices:
        if choice in MODEL_CONFIGS:
            model_names.add(MODEL_CONFIGS[choice].name)

    if not model_names:
        log.warning("No models selected, defaulting to local models.")
        model_names.update(
            config.name
            for config in MODEL_CONFIGS.values()
            if config.provider == "local"
        )
    return sorted(list(model_names))


def _get_completion_model_names(args) -> list[str | None]:
    """Resolves completion model selection from command-line arguments."""
    llm_choices = set(args.llms)
    if "none" in llm_choices:
        return [None]

    llm_names = set()
    if "all" in llm_choices:
        llm_names.update(COMPLETION_MODEL_CONFIGS.keys())
    else:
        llm_names.update(c for c in llm_choices if c in COMPLETION_MODEL_CONFIGS)

    if not llm_names:
        log.warning("No valid LLMs selected, skipping completion step.")
        return [None]

    return sorted(list(llm_names))


def plan_benchmark_jobs(args, run_id: str) -> list[BenchmarkJob]:
    """
    Generates a complete list of benchmark jobs based on command-line arguments.
    """
    documents = _get_documents(args)
    if not documents:
        return []

    db_providers = _get_db_providers(args)
    model_names = _get_model_names(args)
    completion_models = _get_completion_model_names(args)

    log.info(f"Using databases: {db_providers}")
    log.info(f"Using embedding models: {model_names}")
    log.info(f"Using completion models: {completion_models}")

    # Validate S3 configuration if using S3 storage mode
    if args.storage_mode == "s3":
        app_config.validate_s3_config()

    jobs = []
    job_combinations = product(
        documents, db_providers, model_names, completion_models, range(args.runs)
    )

    for i, (doc_info, db_name, model_name, completion_model, run_num) in enumerate(
        job_combinations
    ):
        job_id = f"{run_id}-{i}"
        model_name_safe = model_name.replace("/", "_").replace(":", "_")
        doc_name_safe = doc_info.name.replace("/", "_")

        if args.storage_mode == "s3":
            storage_path = f"s3://{app_config.s3_bucket}/databases/{run_id}/{db_name}/{model_name_safe}/{doc_name_safe}"
        else:
            storage_path = os.path.join(
                tempfile.gettempdir(),
                "vector_db_benchmark",
                run_id,
                db_name,
                model_name_safe,
                doc_name_safe,
            )
            os.makedirs(os.path.dirname(storage_path), exist_ok=True)

        jobs.append(
            BenchmarkJob(
                run_id=run_id,
                job_id=job_id,
                run_number=run_num + 1,
                doc_info=doc_info,
                embedding_model=model_name,
                vector_db_name=db_name,
                db_storage_path=storage_path,
                results_path=args.results_dir,
                completion_model=completion_model,
                cleanup=(not args.no_cleanup),
                storage_mode=args.storage_mode,
            )
        )
    return jobs


def extract_text_from_doc(doc_info: DocumentInfo) -> str:
    """Extracts all text content from a document."""
    processor = PDFProcessor()
    return processor.extract_text(doc_info.path)


def generate_queries_from_text(text: str, num_queries: int = 1) -> list[str]:
    """Generates a specified number of search queries from the text."""
    if not num_queries:
        return []

    query_generator = QueryGenerator()
    queries = query_generator.generate_queries(text, num_queries=num_queries)
    log.info(f"   ✓ Generated {len(queries)} queries")
    return queries


def embed_document(
    text: str, doc_name: str, embedder: EmbeddingProvider
) -> list[dict[str, Any]]:
    """Chunks and embeds document text."""
    document_metadata = {"document_name": doc_name}
    chunks = embedder.embed_document(text, document_metadata)
    log.info(f"   ✓ Model: {embedder.config.model_name}")
    log.info(f"   ✓ Dimensions: {embedder.embedding_dimension}")
    log.info(f"   ✓ Created {len(chunks)} chunks")
    return chunks


def create_and_populate_db(
    db: VectorDatabase, chunks_with_embeddings: list[dict], collection_name: str
) -> None:
    """Creates a new collection and inserts chunks."""
    db.create_collection(collection_name, len(chunks_with_embeddings[0]["embedding"]))
    db.insert_chunks(chunks_with_embeddings)
    log.info(f"   ✓ Collection: {collection_name}")


def embed_query(query: str, embedder: EmbeddingProvider) -> list[float]:
    """Embeds a single query."""
    return embedder.embed_texts([query])[0]


def embed_queries_batch(
    queries: list[str], embedder: EmbeddingProvider
) -> tuple[list[list[float]], float]:
    """Embeds multiple queries in a batch."""
    if not queries:
        return [], 0.0

    log.info(f"   ⚡ Batch embedding {len(queries)} queries")
    import time

    start_time = time.time()

    try:
        query_embeddings = embedder.embed_texts(queries)
        embedding_time = time.time() - start_time

        log.info(f"   ✓ Batch embedded {len(queries)} queries in {embedding_time:.3f}s")
        return query_embeddings, embedding_time

    except Exception as e:
        log.warning(
            f"Batch embedding failed: {e}. Falling back to individual embedding."
        )

        query_embeddings = []
        for i, query in enumerate(queries):
            log.info(f"      Embedding query {i+1}/{len(queries)}")
            embedding = embed_query(query, embedder)
            query_embeddings.append(embedding)

        embedding_time = time.time() - start_time
        log.info(f"   ✓ Fallback completed in {embedding_time:.3f}s")
        return query_embeddings, embedding_time


def run_search(
    db: VectorDatabase, query_embedding: list[float], top_k: int
) -> list[tuple[float, str, dict]]:
    """Runs a single search against the database."""
    return db.search(query_embedding, top_k=top_k)
