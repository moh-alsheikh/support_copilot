"""Optional OCR helpers for scanned PDFs and image files.

`pytesseract` is a Python wrapper; the Tesseract program must also be installed on the computer.
PDF OCR additionally uses Poppler through `pdf2image`. Text, Word, CSV, and JSON ingestion keep
working when these optional system tools are absent.
"""

from functools import lru_cache
from pathlib import Path
import shutil
import subprocess

AVAILABLE = "available"
MISSING = "missing"
BROKEN = "broken"


class OCRDependencyError(RuntimeError):
    """Raised when OCR was requested but a required optional tool is unavailable."""


def clean_ocr_text(text: str) -> str:
    """Remove empty OCR lines and collapse repeated whitespace inside each line."""

    return "\n".join(" ".join(line.split()) for line in text.splitlines() if line.strip())


def _python_package_status(module: str) -> str:
    try:
        __import__(module)
        return AVAILABLE
    except ImportError:
        return MISSING


@lru_cache(maxsize=None)
def _program_status(*commands: tuple[str, ...]) -> str:
    """Return available/missing/broken for the first of `commands` that is on PATH and runs.

    Being on PATH is not enough: a binary whose shared libraries were removed by a package
    upgrade is found by `shutil.which` but aborts on launch, so each candidate is actually run.
    """

    found = [command for command in commands if shutil.which(command[0])]
    if not found:
        return MISSING
    for command in found:
        try:
            if subprocess.run(command, capture_output=True, timeout=10).returncode == 0:
                return AVAILABLE
        except (OSError, subprocess.TimeoutExpired):
            continue
    return BROKEN


def check_ocr_dependencies() -> dict[str, str]:
    """Return `available`, `missing`, or `broken` for each OCR Python and system dependency."""

    return {
        "pytesseract_python": _python_package_status("pytesseract"),
        "tesseract_program": _program_status(("tesseract", "--version")),
        "pdf2image_python": _python_package_status("pdf2image"),
        "poppler_program": _program_status(("pdftoppm", "-v"), ("pdftocairo", "-v")),
    }


def _require_program(status: str, name: str) -> None:
    if status == MISSING:
        raise OCRDependencyError(f"Install the {name} program. See OCR_SETUP.md.")
    if status == BROKEN:
        raise OCRDependencyError(
            f"{name} is installed but fails to run (often a broken shared library after a package "
            "upgrade). Reinstall it. See OCR_SETUP.md."
        )


def _require_image_ocr() -> None:
    """Raise a beginner-readable error when image OCR dependencies are incomplete."""

    status = check_ocr_dependencies()
    if status["pytesseract_python"] != AVAILABLE:
        raise OCRDependencyError("Install pytesseract with: pip install -r requirements.txt")
    _require_program(status["tesseract_program"], "Tesseract")


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
    if status["pdf2image_python"] != AVAILABLE:
        raise OCRDependencyError("Install pdf2image with: pip install -r requirements.txt")
    _require_program(status["poppler_program"], "Poppler")

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
