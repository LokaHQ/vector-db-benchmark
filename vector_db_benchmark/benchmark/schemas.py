from dataclasses import dataclass, field
from typing import Any

from vector_db_benchmark.utils.document_manager import DocumentInfo


@dataclass
class BenchmarkJob:
    """Configuration for a single, self-contained benchmark run."""

    # Core identification
    run_id: str
    job_id: str
    run_number: int

    # Document and model configuration
    doc_info: DocumentInfo
    embedding_model: str
    vector_db_name: str

    # Configuration for persistence
    db_storage_path: str
    results_path: str

    # Optional model configuration
    completion_model: str | None = None

    # Flags and settings
    cleanup: bool = True
    generate_queries: bool = True
    top_k: int = 5
    storage_mode: str = "local"

    # Placeholders for results
    ingestion_result: dict[str, Any] = field(default_factory=dict)
    search_results: list[dict[str, Any]] = field(default_factory=list)
