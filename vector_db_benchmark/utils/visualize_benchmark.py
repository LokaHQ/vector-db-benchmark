import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from vector_db_benchmark.utils.logging_config import log

from ..utils.metrics import BenchmarkFile


def load_ingestion_data(ingestion_file: str) -> pd.DataFrame:
    """Loads ingestion benchmark data."""
    data = BenchmarkFile.load(ingestion_file)
    return pd.DataFrame(data.get("results", []))


def print_summary(df: pd.DataFrame):
    """Prints a textual summary grouped by embedding model and DB."""
    grouping_cols = ["embedding_model", "vector_db_type"]
    metrics = [
        "pdf_processing_time",
        "embedding_time",
        "insertion_time",
        "total_time",
    ]
    summary = (
        df.groupby(grouping_cols)[metrics]
        .mean()
        .round(2)
        .reset_index()
        .sort_values("total_time")
    )
    log.info("Average timings (seconds) per model & DB:")
    log.info(summary.to_string(index=False))

    def size_bucket(size_mb: float) -> str:
        if size_mb < 2:
            return "small (<2MB)"
        if size_mb < 10:
            return "medium (2–10MB)"
        return "large (10+MB)"

    if "document_size_mb" in df.columns:
        df["size_bucket"] = df["document_size_mb"].apply(size_bucket)
        size_summary = (
            df.groupby("size_bucket")["total_time"]
            .mean()
            .round(2)
            .reset_index()
            .rename(columns={"total_time": "avg_total_time_s"})
        )
        log.info("Average total_time by document size bucket:")
        log.info(size_summary.to_string(index=False))


def bar_chart(df: pd.DataFrame, output: Path | None = None):
    """Creates a bar chart of average total_time."""
    pivot = (
        df.groupby(["vector_db_type", "embedding_model"])["total_time"]
        .mean()
        .unstack(fill_value=0)
    )
    pivot.plot(kind="bar", figsize=(8, 4))
    plt.ylabel("Avg total_time (s)")
    plt.title("Benchmark – Average Total Processing Time")
    plt.tight_layout()
    if output:
        plt.savefig(output, dpi=150)
        log.info(f"Chart saved to {output}")
    else:
        plt.show()


def main(argv: list[str] | None = None):
    """Visualizes benchmark results."""
    parser = argparse.ArgumentParser(description="Visualize benchmark results")
    parser.add_argument(
        "--file",
        default="data/benchmark_results/ingestion_results.json",
        help="Path to ingestion_results.json",
    )
    parser.add_argument(
        "--no-chart",
        action="store_true",
        help="Skip showing/saving the bar chart",
    )
    parser.add_argument(
        "--save-chart",
        metavar="PATH",
        help="Save the bar chart instead of showing it",
    )
    args = parser.parse_args(argv)

    file_path = Path(args.file)
    if not file_path.exists():
        parser.error(f"File not found: {file_path}")

    df = load_ingestion_data(args.file)
    if df.empty:
        log.info("No individual runs found in the results file.")
        return

    print_summary(df)

    if not args.no_chart:
        bar_chart(df, Path(args.save_chart) if args.save_chart else None)


if __name__ == "__main__":
    main()
