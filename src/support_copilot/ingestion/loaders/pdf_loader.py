"""Load PDF pages with text extraction first and optional OCR fallback.

Mixed PDFs are handled one page at a time. A page with a useful text layer stays on the fast
`pypdf` path; only an empty or extremely short page is rendered and sent to Tesseract.
"""

from pathlib import Path

from .base import LoadResult, build_record
from ..ocr import OCRDependencyError, ocr_pdf_page


def load_pdf(
    path: str | Path,
    source: str,
    *,
    ocr_enabled: bool = True,
    min_text_chars: int = 40,
    ocr_dpi: int = 300,
    ocr_language: str = "eng",
) -> LoadResult:
    """Extract every readable PDF page and preserve its one-based page number."""

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Install pypdf with: pip install -r requirements.txt") from exc

    result = LoadResult()
    reader = PdfReader(str(path))
    for page_number, page in enumerate(reader.pages, start=1):
        extracted = (page.extract_text() or "").strip()
        text = extracted
        extraction_method = "pypdf"
        document_type = "text_pdf"

        if len(extracted) < min_text_chars and ocr_enabled:
            try:
                ocr_text = ocr_pdf_page(path, page_number, dpi=ocr_dpi, language=ocr_language)
                if len(ocr_text) > len(extracted):
                    text = ocr_text
                    extraction_method = "ocr"
                    document_type = "scanned_pdf"
            except OCRDependencyError as exc:
                result.warnings.append(f"Page {page_number}: OCR skipped. {exc}")
            except Exception as exc:
                result.warnings.append(f"Page {page_number}: OCR failed: {exc}")

        if text:
            result.records.append(
                build_record(
                    text,
                    source=source,
                    location=f"page:{page_number}",
                    document_type=document_type,
                    extraction_method=extraction_method,
                    page=page_number,
                )
            )
        else:
            result.warnings.append(f"Page {page_number}: no readable text was extracted.")
    return result
