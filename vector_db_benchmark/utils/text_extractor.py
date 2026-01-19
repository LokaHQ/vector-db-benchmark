import io

import PyPDF2

from vector_db_benchmark.services.storage import FileManager
from vector_db_benchmark.utils.logging_config import log


class PDFProcessor:
    """Extracts text from PDF files using a FileManager for I/O."""

    def __init__(self):
        self.file_manager = FileManager()

    def extract_text(self, file_path: str) -> str:
        """
        Extracts all text content from a PDF file.

        Args:
            file_path: The path to the PDF file (local or S3 URI).

        Returns:
            The extracted text as a single string.
        """
        log.debug(f"Extracting text from {file_path}")
        try:
            pdf_bytes = self.file_manager.read_bytes(file_path)

            with io.BytesIO(pdf_bytes) as f:
                reader = PyPDF2.PdfReader(f)
                return "".join(
                    page.extract_text() for page in reader.pages if page.extract_text()
                )
        except FileNotFoundError:
            log.error(f"File not found at path: {file_path}")
            return ""
        except Exception as e:
            log.error(f"Failed to process PDF from path {file_path}: {e}")
            return ""
