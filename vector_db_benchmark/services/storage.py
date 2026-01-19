import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from vector_db_benchmark.databases.base import VectorDatabase
from vector_db_benchmark.databases.config import DatabaseConfig
from vector_db_benchmark.services.aws import AWSClient, parse_s3_path
from vector_db_benchmark.utils.logging_config import log


@dataclass
class FileInfo:
    """A simple container for file metadata."""

    path: str
    size_bytes: int


class FileManager:
    """Provides a unified API for file operations on local or S3 paths."""

    def __init__(self, aws_client: AWSClient | None = None):
        """Initializes the FileManager, allowing for AWS client injection for tests."""
        self._aws_client = aws_client or AWSClient()

    def read_bytes(self, path: str) -> bytes:
        """Reads raw bytes from a local or S3 path, raising FileNotFoundError if it doesn't exist."""
        log.debug(f"Reading bytes from: {path}")
        if path.startswith("s3://"):
            content = self._aws_client.read_s3_object_bytes(path)
            if content is None:
                raise FileNotFoundError(f"File not found at S3 path: {path}")
            return content
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found at local path: {path}")
        with open(path, "rb") as f:
            return f.read()

    def write_bytes(self, path: str, content: bytes) -> None:
        """Writes raw bytes to a local or S3 path."""
        log.debug(f"Writing {len(content)} bytes to: {path}")
        if path.startswith("s3://"):
            self._aws_client.write_s3_object_bytes(path, content)
        else:
            # Ensure the local directory exists before writing
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                f.write(content)

    def list_directory(self, path: str) -> list[FileInfo]:
        """Lists file info in a local directory or S3 prefix."""
        log.debug(f"Listing directory: {path}")
        if path.startswith("s3://"):
            bucket, prefix = parse_s3_path(path)
            s3_objects = self._aws_client.list_s3_prefix(bucket, prefix)
            return [
                FileInfo(path=f"s3://{bucket}/{obj['Key']}", size_bytes=obj["Size"])
                for obj in s3_objects
                if not obj["Key"].endswith("/")
            ]
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Directory not found at local path: {path}")

        file_infos = []
        for filename in os.listdir(path):
            full_path = os.path.join(path, filename)
            if os.path.isfile(full_path):
                file_infos.append(
                    FileInfo(path=full_path, size_bytes=os.path.getsize(full_path))
                )
        return file_infos

    def path_exists(self, path: str) -> bool:
        """Checks if a file or directory exists at a local or S3 path."""
        log.debug(f"Checking existence of: {path}")
        if path.startswith("s3://"):
            content = self._aws_client.read_s3_object_bytes(path)
            return content is not None
        return os.path.exists(path)

    def save_directory_to_archive(self, local_dir_path: str, archive_path: str) -> int:
        """
        Creates a gzipped tarball of a directory and saves it to a local or S3 path.
        Returns the size of the created archive in bytes.
        """
        # Create a temporary file to build the archive.
        with tempfile.NamedTemporaryFile(delete=True, suffix=".tar.gz") as tmp_tar_file:
            archive_path_base = tmp_tar_file.name.replace(".tar.gz", "")
            shutil.make_archive(
                base_name=archive_path_base,
                format="gztar",
                root_dir=local_dir_path,
            )
            archive_bytes = Path(f"{archive_path_base}.tar.gz").read_bytes()

        # Ensure the final path has the .tar.gz extension and save it.
        if not archive_path.endswith(".tar.gz"):
            archive_path += ".tar.gz"
        self.write_bytes(archive_path, archive_bytes)
        return len(archive_bytes)

    def load_directory_from_archive(self, archive_path: str, local_dir_path: str):
        """
        Loads a gzipped tarball from a local or S3 path and extracts it.
        """
        # Ensure the source path has the .tar.gz extension and read the bytes.
        if not archive_path.endswith(".tar.gz"):
            archive_path += ".tar.gz"
        archive_bytes = self.read_bytes(archive_path)

        # Write the archive bytes to a temporary file for extraction.
        with tempfile.NamedTemporaryFile(delete=True, suffix=".tar.gz") as tmp_tar_file:
            tmp_tar_file.write(archive_bytes)
            tmp_tar_file.flush()
            shutil.unpack_archive(tmp_tar_file.name, local_dir_path, "gztar")

    def delete_directory(self, path: str):
        """Recursively deletes a directory or a collection of S3 objects with a common prefix."""
        log.debug(f"Deleting directory at path: {path}")
        if path.startswith("s3://"):
            bucket, prefix = parse_s3_path(path)
            self._aws_client.delete_objects_by_prefix(bucket, prefix)
        else:
            if os.path.exists(path):
                try:
                    # For local paths, we assume it's a directory to be removed.
                    shutil.rmtree(path)
                    log.info(f"Successfully deleted local directory: {path}")
                except OSError as e:
                    log.error(f"Error deleting local directory {path}: {e}")


def save_database(db: VectorDatabase, storage_path: str) -> tuple[int, float]:
    """Saves a vector database to a specified path by orchestrating the archival."""
    if not db.requires_persistence():
        log.debug(f"Skipping save for {db.name} - already persistent in cloud")
        return 0, 0.0

    file_manager = FileManager()

    if db.supports_direct_serialization():
        start_time = time.time()
        data_bytes = db.serialize_to_bytes()
        file_manager.write_bytes(storage_path, data_bytes)
        duration = time.time() - start_time
        return len(data_bytes), duration

    with tempfile.TemporaryDirectory() as tmpdir:
        db.save(tmpdir)
        start_time = time.time()
        size_bytes = file_manager.save_directory_to_archive(tmpdir, storage_path)
        duration = time.time() - start_time
    return size_bytes, duration


def load_database(
    db_class: type[VectorDatabase], storage_path: str, config: DatabaseConfig
) -> tuple[VectorDatabase, float]:
    """Loads a vector database from a specified path by orchestrating the extraction."""
    db_instance = db_class(config)
    if not db_instance.requires_persistence():
        log.debug(
            f"Skipping load for {db_instance.name} - reconnecting to cloud storage"
        )
        return db_instance, 0.0

    file_manager = FileManager()

    if (
        hasattr(db_class, "supports_direct_serialization")
        and db_instance.supports_direct_serialization()
    ):
        try:
            start_time = time.time()
            data_bytes = file_manager.read_bytes(storage_path)
            db_instance.deserialize_from_bytes(data_bytes)
            duration = time.time() - start_time
            return db_instance, duration
        except Exception as e:
            log.warning(
                f"Direct deserialization failed, falling back to archive method: {e}"
            )

    load_dir_obj = tempfile.TemporaryDirectory()
    load_dir_path = load_dir_obj.name

    start_time = time.time()
    file_manager.load_directory_from_archive(storage_path, load_dir_path)
    duration = time.time() - start_time

    db_instance = db_class.load(load_dir_path, config)
    db_instance._temp_dir = load_dir_obj

    return db_instance, duration
