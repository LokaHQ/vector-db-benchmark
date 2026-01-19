import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from vector_db_benchmark.config import config
from vector_db_benchmark.services.aws import parse_s3_path
from vector_db_benchmark.services.storage import FileManager
from vector_db_benchmark.utils.logging_config import log


class ResultsConsolidator:
    """Consolidates individual benchmark result files from local or S3 paths into dataframes."""

    def __init__(self, results_base_path: str | None = None):
        """
        Initialize the consolidator with results path.

        Args:
            results_base_path: Path to benchmark results (local path or S3 URI).
                              Defaults to local results directory.
        """
        self.results_base_path = results_base_path or config.get_default_results_dir()
        self.file_manager = FileManager()
        self.is_s3 = self.results_base_path.startswith("s3://")

        # Patterns to match result files
        self.ingestion_pattern = re.compile(r"^ingestion_.*\.json$")
        self.search_pattern = re.compile(r"^search_.*\.json$")

    def list_all_runs(self) -> list[str]:
        """List all benchmark run directories from local or S3 paths."""
        log.info(f"Scanning for benchmark runs in: {self.results_base_path}")

        try:
            # List all items in the results directory
            file_infos = self.file_manager.list_directory(self.results_base_path)

            # Extract unique run directories from file paths
            runs = set()

            if self.is_s3:
                # S3 path handling
                _, base_prefix = parse_s3_path(self.results_base_path)

                for file_info in file_infos:
                    # Extract the run directory from the S3 key
                    _, key = parse_s3_path(file_info.path)
                    # Remove base prefix to get relative path
                    relative_key = key[len(base_prefix):].lstrip("/")

                    # Extract run directory (first part of path)
                    if "/" in relative_key:
                        run_dir = relative_key.split("/")[0]
                        if run_dir.startswith("run_"):
                            runs.add(run_dir)
            else:
                # Local filesystem handling
                base_path = Path(self.results_base_path)

                for file_info in file_infos:
                    # Get relative path from base
                    file_path = Path(file_info.path)
                    try:
                        relative_path = file_path.relative_to(base_path)
                        # Get the first part (run directory)
                        if relative_path.parts:
                            run_dir = relative_path.parts[0]
                            if run_dir.startswith("run_"):
                                runs.add(run_dir)
                    except ValueError:
                        # Path is not relative to base_path, skip it
                        continue

            runs_list = sorted(list(runs))
            log.info(f"Found {len(runs_list)} benchmark runs: {runs_list}")
            return runs_list

        except Exception as e:
            log.error(f"Failed to list benchmark runs: {e}")
            return []

    def get_run_result_files(self, run_id: str) -> dict[str, list[str]]:
        """Get all result files for a specific run."""
        # Construct path consistently for both S3 and local
        if self.is_s3:
            run_path = f"{self.results_base_path.rstrip('/')}/{run_id}"
        else:
            run_path = str(Path(self.results_base_path) / run_id)

        try:
            file_infos = self.file_manager.list_directory(run_path)

            ingestion_files = []
            search_files = []

            for file_info in file_infos:
                filename = Path(file_info.path).name

                if self.ingestion_pattern.match(filename):
                    ingestion_files.append(file_info.path)
                elif self.search_pattern.match(filename):
                    search_files.append(file_info.path)

            log.info(
                f"Run {run_id}: {len(ingestion_files)} ingestion, {len(search_files)} search files"
            )
            return {
                "ingestion": sorted(ingestion_files),
                "search": sorted(search_files),
            }

        except Exception as e:
            log.error(f"Failed to get files for run {run_id}: {e}")
            return {"ingestion": [], "search": []}

    def load_result_file(self, file_path: str) -> dict[str, Any] | None:
        """Load and parse a single result file."""
        try:
            content_bytes = self.file_manager.read_bytes(file_path)
            content_str = content_bytes.decode("utf-8")
            data = json.loads(content_str)

            # Extract the result part (skip system_info)
            return data.get("result", {})

        except Exception as e:
            log.error(f"Failed to load result file {file_path}: {e}")
            return None

    def consolidate_ingestion_results(
        self, runs_to_process: list[str] = None
    ) -> pd.DataFrame:
        """Consolidate all ingestion results into a single dataframe."""
        log.info("Consolidating ingestion results...")

        all_results = []
        runs = runs_to_process or self.list_all_runs()

        for run_id in runs:
            log.info(f"Processing ingestion results for run: {run_id}")
            run_files = self.get_run_result_files(run_id)

            for file_path in run_files["ingestion"]:
                result = self.load_result_file(file_path)
                if result:
                    # Add metadata
                    result["source_file"] = Path(file_path).name
                    result["source_run"] = run_id
                    all_results.append(result)

        if not all_results:
            log.warning("No ingestion results found")
            return pd.DataFrame()

        df = pd.DataFrame(all_results)

        # Convert timestamp if present
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        log.info(f"Consolidated {len(df)} ingestion results from {len(runs)} runs")
        return df

    def consolidate_search_results(
        self, runs_to_process: list[str] = None
    ) -> pd.DataFrame:
        """Consolidate all search results into a single dataframe."""
        log.info("Consolidating search results...")

        all_results = []
        runs = runs_to_process or self.list_all_runs()

        for run_id in runs:
            log.info(f"Processing search results for run: {run_id}")
            run_files = self.get_run_result_files(run_id)

            for file_path in run_files["search"]:
                result = self.load_result_file(file_path)
                if result:
                    # Add metadata
                    result["source_file"] = Path(file_path).name
                    result["source_run"] = run_id
                    all_results.append(result)

        if not all_results:
            log.warning("No search results found")
            return pd.DataFrame()

        df = pd.DataFrame(all_results)

        # Convert timestamp if present
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        log.info(f"Consolidated {len(df)} search results from {len(runs)} runs")
        return df

    def export_dataframes(
        self,
        ingestion_df: pd.DataFrame,
        search_df: pd.DataFrame,
        output_dir: str,
        format: str = "csv",
    ) -> None:
        """Export consolidated dataframes to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

        if format.lower() == "csv":
            if not ingestion_df.empty:
                ingestion_file = output_path / f"consolidated_ingestion_{timestamp}.csv"
                ingestion_df.to_csv(ingestion_file, index=False)
                log.info(
                    f"Exported {len(ingestion_df)} ingestion records to: {ingestion_file}"
                )

            if not search_df.empty:
                search_file = output_path / f"consolidated_search_{timestamp}.csv"
                search_df.to_csv(search_file, index=False)
                log.info(f"Exported {len(search_df)} search records to: {search_file}")

        elif format.lower() == "parquet":
            if not ingestion_df.empty:
                ingestion_file = (
                    output_path / f"consolidated_ingestion_{timestamp}.parquet"
                )
                ingestion_df.to_parquet(ingestion_file, index=False)
                log.info(
                    f"Exported {len(ingestion_df)} ingestion records to: {ingestion_file}"
                )

            if not search_df.empty:
                search_file = output_path / f"consolidated_search_{timestamp}.parquet"
                search_df.to_parquet(search_file, index=False)
                log.info(f"Exported {len(search_df)} search records to: {search_file}")

    def print_summary(
        self, ingestion_df: pd.DataFrame, search_df: pd.DataFrame
    ) -> None:
        """Print a summary of the consolidated results."""
        print("\n" + "=" * 60)
        print("📊 CONSOLIDATED RESULTS SUMMARY")
        print("=" * 60)

        if not ingestion_df.empty:
            print(f"\n🔄 INGESTION RESULTS: {len(ingestion_df)} records")
            print(f"   • Runs: {ingestion_df['source_run'].nunique()}")
            print(
                f"   • Databases: {ingestion_df['vector_db_type'].nunique()} ({list(ingestion_df['vector_db_type'].unique())})"
            )
            print(
                f"   • Models: {ingestion_df['embedding_model'].nunique()} ({list(ingestion_df['embedding_model'].unique())})"
            )
            if "document_name" in ingestion_df.columns:
                print(f"   • Documents: {ingestion_df['document_name'].nunique()}")
            if "total_time" in ingestion_df.columns:
                print(f"   • Avg Total Time: {ingestion_df['total_time'].mean():.2f}s")

        if not search_df.empty:
            print(f"\n🔍 SEARCH RESULTS: {len(search_df)} records")
            print(f"   • Runs: {search_df['source_run'].nunique()}")
            print(
                f"   • Databases: {search_df['vector_db_type'].nunique()} ({list(search_df['vector_db_type'].unique())})"
            )
            print(
                f"   • Embedding Models: {search_df['embedding_model'].nunique()} ({list(search_df['embedding_model'].unique())})"
            )
            if "completion_model" in search_df.columns:
                print(
                    f"   • Completion Models: {search_df['completion_model'].nunique()} ({list(search_df['completion_model'].dropna().unique())})"
                )
            if "total_time" in search_df.columns:
                print(f"   • Avg Total Time: {search_df['total_time'].mean():.2f}s")
            if "completion_latency" in search_df.columns:
                print(
                    f"   • Avg Completion Latency: {search_df['completion_latency'].mean():.2f}s"
                )

        print("\n" + "=" * 60)


def main():
    """Main function to consolidate benchmark results from local or S3 paths."""
    parser = argparse.ArgumentParser(
        description="Consolidate benchmark results from local or S3 storage"
    )
    parser.add_argument(
        "--results-path",
        default=None,
        help="Path to benchmark results (local path or S3 URI). Defaults to local results directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/consolidated_results",
        help="Output directory for consolidated files",
    )
    parser.add_argument(
        "--format", choices=["csv", "parquet"], default="csv", help="Export format"
    )
    parser.add_argument(
        "--runs", nargs="+", help="Specific run IDs to process (default: all)"
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print summary, don't export files",
    )

    args = parser.parse_args()

    # Initialize consolidator
    consolidator = ResultsConsolidator(args.results_path)

    # Process results
    storage_type = "S3" if consolidator.is_s3 else "local"
    log.info(f"Starting {storage_type} results consolidation...")

    ingestion_df = consolidator.consolidate_ingestion_results(args.runs)
    search_df = consolidator.consolidate_search_results(args.runs)

    # Print summary
    consolidator.print_summary(ingestion_df, search_df)

    # Export if requested
    if not args.summary_only:
        consolidator.export_dataframes(
            ingestion_df, search_df, args.output_dir, args.format
        )
        log.info(f"Consolidation complete! Files saved to: {args.output_dir}")
    else:
        log.info("Summary complete (no files exported)")


if __name__ == "__main__":
    main()
