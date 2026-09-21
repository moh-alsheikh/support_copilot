"""Load scanned image files through optional local Tesseract OCR."""

from pathlib import Path

from .base import LoadResult, build_record
from ..ocr import ocr_image


def load_image(path: str | Path, source: str, language: str = "eng") -> LoadResult:
    """Extract one normalized record from an image."""

    text = ocr_image(path, language=language)
    if not text:
        return LoadResult(warnings=["OCR completed but found no readable text."])
    return LoadResult(
        records=[
            build_record(
                text,
                source=source,
                location="image:1",
                document_type="scanned_image",
                extraction_method="ocr",
                page=1,
            )
        ]
    )
