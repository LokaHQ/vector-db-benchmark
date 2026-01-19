import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from vector_db_benchmark.utils.logging_config import log


def load_results_file(file_path: Path) -> pd.DataFrame:
    """Loads results from an ingestion or search file into a DataFrame."""
    if not file_path.exists():
        return pd.DataFrame()

    with file_path.open() as f:
        data = json.load(f)

    df = pd.DataFrame(data.get("results", []))

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df


def create_search_results_dataframe(search_df: pd.DataFrame) -> pd.DataFrame:
    """Creates a flat dataframe for individual search results."""
    if search_df.empty:
        return pd.DataFrame()

    rows = []
    for _, search_row in search_df.iterrows():
        if not search_row.get("results"):
            continue

        base_fields = {
            "search_timestamp": search_row["timestamp"],
            "vector_db_type": search_row["vector_db_type"],
            "embedding_model": search_row["embedding_model"],
            "run_number": search_row["run_number"],
            "query": search_row["query"],
            "query_index": search_row["query_index"],
            "total_search_time": search_row["total_time"],
        }

        for result in search_row["results"]:
            row = base_fields.copy()
            row.update(
                {
                    "rank": result.get("rank"),
                    "distance": result.get("distance"),
                    "similarity_score": result.get("similarity_score"),
                    "document_name": result.get("document_name"),
                    "chunk_id": result.get("chunk_id"),
                    "chunk_size": result.get("chunk_size"),
                    "result_text": (
                        result.get("text", "")[:200] + "..."
                        if len(result.get("text", "")) > 200
                        else result.get("text", "")
                    ),
                }
            )
            rows.append(row)

    return pd.DataFrame(rows)


def export_dataframes(
    ingestion_df: pd.DataFrame,
    search_df: pd.DataFrame,
    search_results_df: pd.DataFrame,
    output_dir: Path,
    extension: str = "csv",
) -> None:
    """Exports dataframes to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if extension.lower() == "csv":
        if not ingestion_df.empty:
            ingestion_df.to_csv(output_dir / "ingestion_results.csv", index=False)
            log.info(
                f"Exported {len(ingestion_df)} ingestion records to ingestion_results.csv"
            )

        if not search_df.empty:
            search_df.to_csv(output_dir / "search_results.csv", index=False)
            log.info(f"Exported {len(search_df)} search records to search_results.csv")

        if not search_results_df.empty:
            search_results_df.to_csv(
                output_dir / "individual_search_results.csv", index=False
            )
            log.info(
                f"Exported {len(search_results_df)} individual search results to individual_search_results.csv"
            )

    elif extension.lower() == "parquet":
        if not ingestion_df.empty:
            ingestion_df.to_parquet(
                output_dir / "ingestion_results.parquet", index=False
            )
            log.info(
                f"Exported {len(ingestion_df)} ingestion records to ingestion_results.parquet"
            )

        if not search_df.empty:
            search_df.to_parquet(output_dir / "search_results.parquet", index=False)
            log.info(
                f"Exported {len(search_df)} search records to search_results.parquet"
            )

        if not search_results_df.empty:
            search_results_df.to_parquet(
                output_dir / "individual_search_results.parquet", index=False
            )
            log.info(
                f"Exported {len(search_results_df)} individual search results to individual_search_results.parquet"
            )


def print_summary(
    ingestion_df: pd.DataFrame, search_df: pd.DataFrame, search_results_df: pd.DataFrame
) -> None:
    """Prints summary statistics of the exported data."""
    log.info("=== DATA EXPORT SUMMARY ===")

    if not ingestion_df.empty:
        log.info(f"Ingestion Results: {len(ingestion_df)} records")
        log.info(f"  - Vector DBs: {ingestion_df['vector_db_type'].nunique()} types")
        log.info(
            f"  - Embedding Models: {ingestion_df['embedding_model'].nunique()} models"
        )
        log.info(
            f"  - Documents: {ingestion_df['document_name'].nunique()} unique documents"
        )
        log.info(
            f"  - Total processing time range: {ingestion_df['total_time'].min():.2f}s - {ingestion_df['total_time'].max():.2f}s"
        )

    if not search_df.empty:
        log.info(f"Search Results: {len(search_df)} records")
        log.info(f"  - Vector DBs: {search_df['vector_db_type'].nunique()} types")
        log.info(
            f"  - Embedding Models: {search_df['embedding_model'].nunique()} models"
        )
        log.info(f"  - Unique queries: {search_df['query'].nunique()} queries")
        log.info(
            f"  - Search time range: {search_df['total_time'].min():.3f}s - {search_df['total_time'].max():.3f}s"
        )

    if not search_results_df.empty:
        log.info(f"Individual Search Results: {len(search_results_df)} result items")
        log.info(
            f"  - From {search_results_df['search_timestamp'].nunique()} search operations"
        )
        log.info(
            f"  - Documents referenced: {search_results_df['document_name'].nunique()} unique documents"
        )


def main():
    """Exports benchmark data."""
    parser = argparse.ArgumentParser(
        description="Export benchmark results to flat dataframes"
    )
    parser.add_argument(
        "--ingestion-file",
        default="data/benchmark_results/ingestion_results.json",
        help="Path to ingestion results JSON file",
    )
    parser.add_argument(
        "--search-file",
        default="data/benchmark_results/search_results.json",
        help="Path to search results JSON file",
    )
    parser.add_argument(
        "--output-dir",
        default="data/exported_dataframes",
        help="Output directory for exported files",
    )
    parser.add_argument(
        "--extension",
        choices=["csv", "parquet"],
        default="csv",
        help="Export extension (csv or parquet)",
    )
    parser.add_argument(
        "--no-individual-results",
        action="store_true",
        help="Skip exporting individual search results dataframe",
    )

    args = parser.parse_args()

    log.info("Loading benchmark results...")
    ingestion_df = load_results_file(Path(args.ingestion_file))
    search_df = load_results_file(Path(args.search_file))

    search_results_df = pd.DataFrame()
    if not args.no_individual_results:
        search_results_df = create_search_results_dataframe(search_df)

    output_dir = Path(args.output_dir)
    export_dataframes(
        ingestion_df, search_df, search_results_df, output_dir, args.extension
    )

    print_summary(ingestion_df, search_df, search_results_df)

    log.info(f"Export complete! Files saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
