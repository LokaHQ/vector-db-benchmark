from dotenv import load_dotenv

load_dotenv()

import argparse
import time

from vector_db_benchmark.benchmark.runner import run_benchmark_job
from vector_db_benchmark.benchmark.tasks import plan_benchmark_jobs
from vector_db_benchmark.completion.config import COMPLETION_MODEL_CONFIGS
from vector_db_benchmark.config import config
from vector_db_benchmark.embedding.config import MODEL_CONFIGS
from vector_db_benchmark.utils.document_manager import DocumentManager
from vector_db_benchmark.utils.logging_config import log


def create_parser():
    """Creates and configures the argument parser."""
    parser = argparse.ArgumentParser(description="Vector Database Benchmark")

    parser.add_argument(
        "--docs",
        default="sample",
        choices=["sample", "all", "small", "medium", "large"],
        help="Document selection: sample (1 per category), all, or by size category",
    )
    parser.add_argument(
        "--dbs",
        nargs="+",
        default=["all"],
        choices=["chroma", "qdrant", "milvus_lite", "lance", "s3_vectors", "all"],
        help="Vector database(s) to benchmark",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["local"],
        choices=list(MODEL_CONFIGS.keys()) + ["local", "bedrock", "all"],
        help="Embedding model(s) to benchmark",
    )
    parser.add_argument(
        "--llms",
        nargs="+",
        default=["none"],
        choices=list(COMPLETION_MODEL_CONFIGS.keys()) + ["all", "none"],
        help="Completion model(s) to benchmark. Use 'none' to skip (default). Requires AWS Bedrock for other options.",
    )
    parser.add_argument(
        "--runs", type=int, default=1, help="Number of runs per document"
    )
    parser.add_argument(
        "--storage-mode",
        default="local",
        choices=["local", "s3"],
        help="Storage mode for the vector database: local (in-memory) or s3 (persistent)",
    )
    parser.add_argument(
        "--docs-dir",
        default=config.get_default_docs_dir(),
        help="Documents directory (local path or S3 URI like s3://bucket/path)",
    )
    parser.add_argument(
        "--list-docs", action="store_true", help="List available documents and exit"
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Disable deletion of database artifacts after each run.",
    )
    parser.add_argument(
        "--results-dir",
        default=config.get_default_results_dir(),
        help="Directory to save benchmark result files (local path or S3 URI like s3://bucket/path)",
    )

    return parser


def main(args: argparse.Namespace | None = None) -> tuple[str | None, int]:
    """Runs the benchmark."""
    if args is None:
        parser = create_parser()
        args = parser.parse_args()

    if args.list_docs:
        log.info(f"Listing documents from: {args.docs_dir}")
        try:
            DocumentManager(args.docs_dir)
        except FileNotFoundError as e:
            log.error(e)
        return None, 0

    run_id = f"run_{int(time.time())}"
    log.info(f"Starting Benchmark Run ID: {run_id}")

    all_jobs = plan_benchmark_jobs(args, run_id)

    if not all_jobs:
        log.warning("No benchmark jobs were planned. Exiting.")
        return run_id, 0

    for job in all_jobs:
        run_benchmark_job(job)

    log.info(f"Benchmark Run ID: {run_id} completed.")
    return run_id, len(all_jobs)


if __name__ == "__main__":
    main()
