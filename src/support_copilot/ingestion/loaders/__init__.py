"""Format-specific source loaders reused across the ingestion pipeline.

Each loader turns one source file into the same small record shape. The rest of the RAG pipeline
therefore does not need separate chunking or indexing code for PDF, Word, text, CSV, or JSON.
"""

from .base import LoadResult, build_record

__all__ = ["LoadResult", "build_record"]
