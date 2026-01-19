import json
import platform
import subprocess
import threading
import time
from datetime import datetime
from typing import Any, Literal

import psutil
from pydantic import BaseModel, Field

from vector_db_benchmark.services.storage import FileManager
from vector_db_benchmark.utils.logging_config import log


class SystemInfo(BaseModel):
    """System information."""

    os_name: str
    os_version: str
    python_version: str
    cpu_count: int
    total_ram_gb: float
    cpu_brand: str


class BenchmarkResult(BaseModel):
    """Metrics for ingestion and search operations."""

    # Discriminator field
    type: Literal["ingestion", "search"]
    timestamp: datetime = Field(default_factory=datetime.now)

    # Common fields
    vector_db_type: str
    embedding_model: str
    embedding_dimension: int
    run_number: int | None = None
    storage_mode: str | None = None

    # System metrics
    peak_ram_mb: float
    avg_cpu_percent: float

    # Ingestion-specific fields
    document_name: str | None = None
    document_size_mb: float | None = None
    chunks_created: int | None = None
    text_extraction_time: float | None = None
    query_generation_time: float | None = None
    embedding_time: float | None = None
    insertion_time: float | None = None
    persisted_db_size_bytes: int | None = None

    # Search-specific fields
    query: str | None = None
    query_length: int | None = None
    top_k_requested: int | None = None
    results_returned: int | None = None
    db_load_time: float | None = None
    query_embedding_time: float | None = None
    search_time: float | None = None
    results: list[dict[str, Any]] | None = None

    # RAG Completion Fields
    completion_model: str | None = None
    completion_answer: str | None = None
    completion_latency: float | None = None
    completion_input_tokens: int | None = None
    completion_output_tokens: int | None = None

    # Common timing field
    total_time: float


class BenchmarkMetrics:
    """Handles metrics collection and system monitoring."""

    def __init__(self, background_monitoring: bool = True):
        self.process = psutil.Process()
        self.background_monitoring = background_monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        self.start_ram = 0.0
        self.peak_ram = 0.0
        self.cpu_samples = []

    def start_monitoring(self, background: bool = None):
        """Starts monitoring system resources."""
        background = (
            background if background is not None else self.background_monitoring
        )

        self.start_ram = self.process.memory_info().rss / (1024**2)
        self.peak_ram = self.start_ram
        self.cpu_samples = []

        if background:
            self._start_background_monitoring()

    def _start_background_monitoring(self):
        """Starts the background monitoring thread."""
        self.monitoring_active = True

        def monitor_loop():
            while self.monitoring_active:
                self.update_monitoring()
                time.sleep(0.1)

        self.monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitoring_thread.start()

    def stop_monitoring(self):
        """Stops background monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)

    def update_monitoring(self):
        """Updates resource monitoring data."""
        current_ram = self.process.memory_info().rss / (1024**2)
        self.peak_ram = max(self.peak_ram, current_ram)

        cpu_percent = self.process.cpu_percent()
        if cpu_percent > 0:
            self.cpu_samples.append(cpu_percent)

    def get_resource_stats(self) -> dict[str, float]:
        """Gets current resource usage statistics."""
        avg_cpu = (
            sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0
        )
        return {"peak_ram_mb": self.peak_ram, "avg_cpu_percent": avg_cpu}


def collect_system_info() -> SystemInfo:
    """Collects static system information."""
    cpu_brand = "Unknown"
    try:
        if platform.system() == "Darwin":
            result = subprocess.run(
                ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                cpu_brand = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass

    return SystemInfo(
        os_name=platform.system(),
        os_version=platform.release(),
        python_version=platform.python_version(),
        cpu_count=psutil.cpu_count(),
        total_ram_gb=psutil.virtual_memory().total / (1024**3),
        cpu_brand=cpu_brand,
    )


def save_benchmark_result(
    result_type: str,
    metrics_data: dict[str, Any],
    output_path: str,
) -> None:
    """Saves a single benchmark result as a separate JSON file."""
    # Extract IDs for constructing a unique path
    run_id = metrics_data.get("run_id", "unknown_run")
    job_id = metrics_data.get("job_id", "unknown_job")

    file_path = f"{output_path}/{run_id}/{result_type}_{job_id}.json"

    # Create the data models
    system_info = collect_system_info()
    result_model = BenchmarkResult(type=result_type, **metrics_data)

    # Structure the final output for the JSON file
    final_output = {
        "system_info": system_info.model_dump(),
        "result": result_model.model_dump(),
    }

    # Serialize to JSON bytes
    json_content = json.dumps(final_output, indent=2, default=str).encode("utf-8")

    # Save using the FileManager
    try:
        file_manager = FileManager()
        file_manager.write_bytes(file_path, json_content)
        log.info(f"Successfully saved result to {file_path}")
    except Exception as e:
        log.error(f"Failed to save benchmark result to {file_path}: {e}")
