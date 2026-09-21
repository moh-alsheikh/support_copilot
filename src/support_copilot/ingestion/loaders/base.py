"""Shared types and helpers for document loaders.

A loader has one job: extract readable text and say where that text came from. `LoadResult` also
keeps warnings beside successful records so one imperfect page does not cancel an entire corpus.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LoadResult:
    """Records and non-fatal warnings produced while loading one file."""

    records: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def build_record(
    text: str,
    *,
    source: str,
    location: str,
    document_type: str,
    extraction_method: str,
    page: int | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the normalized source record expected by chunking and indexing."""

    metadata: dict[str, Any] = {
        "source": source,
        "page": page,
        "location": location,
        "document_type": document_type,
        "extraction_method": extraction_method,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return {"text": text.strip(), "metadata": metadata}
