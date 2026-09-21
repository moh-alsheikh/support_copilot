"""Optional OCR helpers for scanned PDFs and image files.

`pytesseract` is a Python wrapper; the Tesseract program must also be installed on the computer.
PDF OCR additionally uses Poppler through `pdf2image`. Text, Word, CSV, and JSON ingestion keep
working when these optional system tools are absent.
"""

from pathlib import Path
import shutil
from typing import Any


class OCRDependencyError(RuntimeError):
    """Raised when OCR was requested but a required optional tool is unavailable."""


def clean_ocr_text(text: str) -> str:
    """Remove empty OCR lines and collapse repeated whitespace inside each line."""

    return "\n".join(" ".join(line.split()) for line in text.splitlines() if line.strip())


def check_ocr_dependencies() -> dict[str, Any]:
    """Return an easy-to-display status report for OCR Python and system dependencies."""

    try:
        import pytesseract  # noqa: F401

        pytesseract_available = True
    except ImportError:
        pytesseract_available = False

    try:
        import pdf2image  # noqa: F401

        pdf2image_available = True
    except ImportError:
        pdf2image_available = False

    return {
        "pytesseract_python": pytesseract_available,
        "tesseract_program": shutil.which("tesseract") is not None,
        "pdf2image_python": pdf2image_available,
        "poppler_program": bool(shutil.which("pdftoppm") or shutil.which("pdftocairo")),
    }


def _require_image_ocr() -> None:
    """Raise a beginner-readable error when image OCR dependencies are incomplete."""

    status = check_ocr_dependencies()
    if not status["pytesseract_python"]:
        raise OCRDependencyError("Install pytesseract with: pip install -r requirements.txt")
    if not status["tesseract_program"]:
        raise OCRDependencyError("Install the Tesseract program. See OCR_SETUP.md.")


def ocr_image(image_path: str | Path, language: str = "eng") -> str:
    """Extract and clean text from one image file."""

    _require_image_ocr()
    from PIL import Image
    import pytesseract

    with Image.open(image_path) as image:
        return clean_ocr_text(pytesseract.image_to_string(image, lang=language))


def ocr_pdf_page(pdf_path: str | Path, page_number: int, dpi: int = 300, language: str = "eng") -> str:
    """Render one one-based PDF page and extract its text with Tesseract."""

    _require_image_ocr()
    status = check_ocr_dependencies()
    if not status["pdf2image_python"]:
        raise OCRDependencyError("Install pdf2image with: pip install -r requirements.txt")
    if not status["poppler_program"]:
        raise OCRDependencyError("Install Poppler so scanned PDF pages can be rendered. See OCR_SETUP.md.")

    from pdf2image import convert_from_path
    import pytesseract

    images = convert_from_path(
        str(pdf_path),
        dpi=dpi,
        first_page=page_number,
        last_page=page_number,
    )
    if not images:
        return ""
    return clean_ocr_text(pytesseract.image_to_string(images[0], lang=language))
