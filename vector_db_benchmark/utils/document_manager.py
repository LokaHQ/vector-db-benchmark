import os
from dataclasses import dataclass

from vector_db_benchmark.services.storage import FileManager
from vector_db_benchmark.utils.logging_config import log


@dataclass
class DocumentInfo:
    """Holds metadata for a single document."""

    name: str
    path: str
    size_bytes: int

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)


class DocumentManager:
    """Discovers and categorizes documents from a local or S3 path."""

    SIZE_CATEGORIES = {
        "small": (0, 2),
        "medium": (2, 10),
        "large": (10, 50),
    }

    def __init__(self, path: str):
        """Initializes the DocumentManager."""
        self.path = path
        self.file_manager = FileManager()
        self.documents = self._load_documents()
        self._categorize_documents()
        self.print_document_summary()

    def _load_documents(self) -> list[DocumentInfo]:
        """Loads documents from the configured path using the FileManager."""
        log.info(f"Discovering documents in: {self.path}")
        try:
            file_infos = self.file_manager.list_directory(self.path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No documents found at path: {self.path}"
            ) from None

        documents = [
            DocumentInfo(
                name=os.path.basename(file_info.path),
                path=file_info.path,
                size_bytes=file_info.size_bytes,
            )
            for file_info in file_infos
            if file_info.path.lower().endswith(".pdf")
        ]

        if not documents:
            raise FileNotFoundError(f"No PDF documents found at path: {self.path}")

        return documents

    def _categorize_documents(self):
        """Categorizes documents by size."""
        self.categories = {"small": [], "medium": [], "large": []}
        for doc in self.documents:
            size_mb = doc.size_bytes / (1024 * 1024)
            category = self._get_size_category(size_mb)
            self.categories[category].append(doc)

        self.documents.sort(key=lambda d: d.size_bytes)

    def _get_size_category(self, size_mb: float) -> str:
        """Determines the size category for a document."""
        if (
            self.SIZE_CATEGORIES["small"][0]
            <= size_mb
            < self.SIZE_CATEGORIES["small"][1]
        ):
            return "small"
        if (
            self.SIZE_CATEGORIES["medium"][0]
            <= size_mb
            < self.SIZE_CATEGORIES["medium"][1]
        ):
            return "medium"
        return "large"

    def get_document_by_name(self, name: str) -> DocumentInfo | None:
        """Retrieves a document by its name."""
        return next((doc for doc in self.documents if doc.name == name), None)

    def get_documents_by_category(self, category: str) -> list[DocumentInfo]:
        """Returns all documents belonging to a specific size category."""
        return self.categories.get(category, [])

    def get_all_documents(self) -> list[DocumentInfo]:
        """Returns all discovered documents."""
        return self.documents

    def get_sample_documents(self, per_category: int = 1) -> list[DocumentInfo]:
        """Returns a small, representative sample of documents."""
        sample_docs = []
        for category in self.categories:
            sample_docs.extend(self.categories[category][:per_category])
        return sorted(sample_docs, key=lambda d: d.size_bytes)

    def print_document_summary(self):
        """Prints a compact summary of available documents."""
        log.info("-" * 50)
        log.info(f"Found {len(self.documents)} documents in {self.path}")
        log.info("-" * 50)

        for category, (min_size, max_size) in self.SIZE_CATEGORIES.items():
            category_name = f"{category.capitalize()} ({min_size}-{max_size}MB)"
            count = len(self.categories.get(category, []))
            log.info(f"  - {category_name:<25}: {count} docs")
        log.info("-" * 50)
