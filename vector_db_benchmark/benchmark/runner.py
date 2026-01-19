import time
from typing import Any

from vector_db_benchmark.benchmark.schemas import BenchmarkJob
from vector_db_benchmark.benchmark.tasks import (
    create_and_populate_db,
    embed_document,
    embed_queries_batch,
    extract_text_from_doc,
    generate_queries_from_text,
    run_search,
)
from vector_db_benchmark.completion.factory import create_completion_provider
from vector_db_benchmark.databases.config import DatabaseConfig
from vector_db_benchmark.databases.factory import get_database_provider
from vector_db_benchmark.embedding.factory import create_embedder
from vector_db_benchmark.services.storage import FileManager, load_database, save_database
from vector_db_benchmark.utils.logging_config import log
from vector_db_benchmark.utils.metrics import BenchmarkMetrics, save_benchmark_result


def time_operation(operation_name: str, operation_func) -> tuple[Any, float]:
    """Times a function and logs the duration."""
    log.info(f"{operation_name}...")
    start_time = time.time()
    result = operation_func()
    duration = time.time() - start_time
    log.info(f"   ✓ {operation_name} completed in {duration:.3f}s")
    return result, duration


def sanitize_collection_name(doc_name: str, timestamp: int) -> str:
    """Creates a valid collection name for all database providers."""
    import re

    clean_name = doc_name.replace(".pdf", "").replace(".txt", "")
    clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", clean_name)
    clean_name = re.sub(r"_+", "_", clean_name)
    clean_name = clean_name.strip("_")
    if clean_name and not clean_name[0].isalpha():
        clean_name = f"doc_{clean_name}"
    return f"{clean_name}_{timestamp}"[:512]


def run_benchmark_job(job: BenchmarkJob) -> dict[str, Any]:
    """Runs a single, self-contained benchmark job."""
    log.info("=" * 80)
    log.info(f"Starting Job: {job.job_id}")
    log.info(f"Document: {job.doc_info.name} ({job.doc_info.size_mb:.1f}MB)")
    log.info(f"DB: {job.vector_db_name}, Model: {job.embedding_model}")
    if job.completion_model:
        log.info(f"LLM: {job.completion_model}")
    log.info("=" * 80)

    file_manager = FileManager()
    metrics_collector = BenchmarkMetrics()
    metrics_collector.start_monitoring()

    embedder = create_embedder(model_name=job.embedding_model)

    db_class = get_database_provider(job.vector_db_name)
    db_config = DatabaseConfig()

    # Ingestion Phase
    ingestion_start_time = time.time()

    text, text_extraction_time = time_operation(
        "1. Extracting text", lambda: extract_text_from_doc(job.doc_info)
    )
    queries, query_generation_time = time_operation(
        "2. Generating queries",
        lambda: generate_queries_from_text(text, num_queries=3),
    )
    chunks, embedding_time = time_operation(
        "3. Embedding document",
        lambda: embed_document(text, job.doc_info.name, embedder),
    )

    with db_class(config=db_config) as db_instance:
        collection_name = sanitize_collection_name(job.doc_info.name, int(time.time()))
        _, insertion_time = time_operation(
            "4. Populating DB",
            lambda: create_and_populate_db(db_instance, chunks, collection_name),
        )

        persisted_size, save_time = save_database(db_instance, job.db_storage_path)

    ingestion_total_time = time.time() - ingestion_start_time

    resource_stats = metrics_collector.get_resource_stats()

    ingestion_result = {
        "job_id": job.job_id,
        "run_id": job.run_id,
        "run_number": job.run_number,
        "storage_mode": job.storage_mode,
        "document_name": job.doc_info.name,
        "document_size_mb": job.doc_info.size_mb,
        "vector_db_type": job.vector_db_name,
        "embedding_model": job.embedding_model,
        "embedding_dimension": embedder.embedding_dimension,
        "chunks_created": len(chunks),
        "text_extraction_time": text_extraction_time,
        "query_generation_time": query_generation_time,
        "embedding_time": embedding_time,
        "insertion_time": insertion_time,
        "save_time": save_time,
        "persisted_db_size_bytes": persisted_size,
        "total_ingestion_time": ingestion_total_time,
        "peak_ram_mb": resource_stats["peak_ram_mb"],
        "avg_cpu_percent": resource_stats["avg_cpu_percent"],
        "total_time": ingestion_total_time,
    }
    save_benchmark_result("ingestion", ingestion_result, job.results_path)

    # Search Phase
    search_start_time = time.time()

    loaded_db_instance, load_time = load_database(
        db_class, job.db_storage_path, db_config
    )
    with loaded_db_instance as loaded_db:
        log.info(f"Running {len(queries)} searches...")

        log.info(f"Batch embedding {len(queries)} queries...")
        query_embeddings, total_embedding_time = embed_queries_batch(queries, embedder)

        avg_query_embedding_time = (
            total_embedding_time / len(queries) if queries else 0.0
        )

        completion_provider = None
        if job.completion_model:
            completion_provider = create_completion_provider(job.completion_model)

        for i, (query, query_embedding) in enumerate(
            zip(queries, query_embeddings, strict=True)
        ):
            log.info(f"Running search {i + 1}/{len(queries)}: '{query[:50]}...'")

            search_results, search_time = time_operation(
                f"Searching {loaded_db.name}",
                lambda query_embedding=query_embedding: run_search(
                    loaded_db, query_embedding, job.top_k
                ),
            )
            log.info(f"   ✓ Found {len(search_results)} results")

            formatted_results = [
                {
                    "rank": j,
                    "similarity_score": score,
                    "text": text,
                    **metadata,
                }
                for j, (score, text, metadata) in enumerate(search_results, 1)
            ]

            total_search_time = avg_query_embedding_time + search_time

            search_result_to_save = {
                "job_id": job.job_id,
                "run_id": job.run_id,
                "run_number": job.run_number,
                "storage_mode": job.storage_mode,
                "document_name": job.doc_info.name,
                "vector_db_type": job.vector_db_name,
                "embedding_model": job.embedding_model,
                "embedding_dimension": embedder.embedding_dimension,
                "db_load_time": load_time,
                "query": query,
                "top_k_requested": job.top_k,
                "results_returned": len(search_results),
                "query_embedding_time": avg_query_embedding_time,
                "search_time": search_time,
                "total_time": total_search_time,
                "results": formatted_results,
                "peak_ram_mb": resource_stats["peak_ram_mb"],
                "avg_cpu_percent": resource_stats["avg_cpu_percent"],
            }

            # Completion Phase
            if completion_provider:
                log.info(f"Running completion for query {i + 1}/{len(queries)}...")
                context_for_llm = [res["text"] for res in formatted_results]

                completion_result, completion_latency = time_operation(
                    f"Generating completion with {job.completion_model}",
                    lambda q=query, c=context_for_llm: completion_provider.complete(
                        q, c
                    ),
                )

                search_result_to_save.update(
                    {
                        "completion_model": job.completion_model,
                        "completion_answer": completion_result.answer,
                        "completion_latency": completion_latency,
                        "completion_input_tokens": completion_result.input_tokens,
                        "completion_output_tokens": completion_result.output_tokens,
                    }
                )

            save_benchmark_result("search", search_result_to_save, job.results_path)

    search_total_time = time.time() - search_start_time

    metrics_collector.stop_monitoring()

    if job.cleanup:
        log.info(f"Cleaning up database artifacts at {job.db_storage_path}...")
        try:
            file_manager.delete_directory(job.db_storage_path)
            log.info("   ✓ Cleanup successful.")
        except Exception as e:
            log.error(f"   ✗ Cleanup failed: {e}")

    log.info(f"Job {job.job_id} completed.")
    log.info(
        f"Ingestion: {ingestion_total_time:.2f}s, Search: {search_total_time:.2f}s"
    )
    log.info(f"Peak RAM: {resource_stats['peak_ram_mb']:.2f} MB")
    log.info("=" * 80)

    return {
        "ingestion_result": ingestion_result,
        "search_total_time": search_total_time,
    }
