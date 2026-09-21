"""Discover and normalize every supported source file under `data/raw/`.

This is the ingestion coordinator. Format-specific details stay in `ingestion/loaders/`; this
module handles recursive discovery, duplicate detection, failure isolation, and document-register
metadata before LlamaIndex receives Documents.
"""

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any

from ..config import AppConfig
from .document_register import ensure_document_register, register_metadata, save_document_register
from .loaders.base import LoadResult
from .loaders.csv_loader import load_csv
from .loaders.docx_loader import load_docx
from .loaders.image_loader import load_image
from .loaders.json_loader import load_json
from .loaders.pdf_loader import load_pdf
from .loaders.text_loader import load_text


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp",
    ".txt", ".md", ".markdown",
    ".docx", ".csv", ".json", ".jsonl",
}
LEGACY_EXTENSIONS = {".doc"}


@dataclass
class FilePreparationStatus:
    """One file's outcome, suitable for terminal output and the Streamlit health tab."""

    source: str
    extension: str
    status: str
    record_count: int = 0
    warnings: list[str] = field(default_factory=list)
    error: str = ""


@dataclass
class PreparationResult:
    """All normalized records and per-file outcomes from one preparation run."""

    records: list[dict[str, Any]] = field(default_factory=list)
    files: list[FilePreparationStatus] = field(default_factory=list)

    def summary(self) -> dict[str, int]:
        """Count files by status for concise reporting."""

        counts: dict[str, int] = {}
        for item in self.files:
            counts[item.status] = counts.get(item.status, 0) + 1
        return counts


def discover_files(raw_dir: str | Path) -> list[Path]:
    """Return non-hidden files recursively, in deterministic relative-path order."""

    root = Path(raw_dir)
    if not root.exists():
        return []
    files = [path for path in root.rglob("*") if path.is_file() and not any(part.startswith(".") for part in path.relative_to(root).parts)]
    return sorted(files, key=lambda path: path.relative_to(root).as_posix().lower())


def file_sha256(path: str | Path) -> str:
    """Calculate a content hash used to identify duplicate source files."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source_file(path: Path, source: str, config: AppConfig) -> LoadResult:
    """Dispatch one supported path to its format-specific loader."""

    extension = path.suffix.lower()
    if extension == ".pdf":
        return load_pdf(
            path,
            source,
            ocr_enabled=config.ocr_enabled,
            min_text_chars=config.ocr_min_chars,
            ocr_dpi=config.ocr_dpi,
            ocr_language=config.ocr_language,
        )
    if extension in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        return load_image(path, source, language=config.ocr_language)
    if extension in {".txt", ".md", ".markdown"}:
        return load_text(path, source)
    if extension == ".docx":
        return load_docx(path, source)
    if extension == ".csv":
        return load_csv(path, source)
    if extension in {".json", ".jsonl"}:
        return load_json(path, source)
    raise ValueError(f"No loader is registered for {extension or 'files without an extension'}.")


def prepare_documents(config: AppConfig) -> PreparationResult:
    """Prepare the complete raw corpus while isolating errors to individual files."""

    raw_dir = config.raw_data_dir
    discovered = discover_files(raw_dir)
    supported = [path for path in discovered if path.suffix.lower() in SUPPORTED_EXTENSIONS]
    hashes = {path.relative_to(raw_dir).as_posix(): file_sha256(path) for path in supported}
    register = ensure_document_register(supported, raw_dir, config.document_register_path, hashes)

    output = PreparationResult()
    seen_hashes: dict[str, str] = {}
    for path in discovered:
        source = path.relative_to(raw_dir).as_posix()
        extension = path.suffix.lower()
        if extension in LEGACY_EXTENSIONS:
            output.files.append(
                FilePreparationStatus(
                    source=source,
                    extension=extension,
                    status="unsupported",
                    warnings=["Legacy .doc is not supported. Export it as .docx and run preparation again."],
                )
            )
            continue
        if extension not in SUPPORTED_EXTENSIONS:
            output.files.append(FilePreparationStatus(source=source, extension=extension, status="unsupported"))
            continue

        content_hash = hashes[source]
        if content_hash in seen_hashes:
            output.files.append(
                FilePreparationStatus(
                    source=source,
                    extension=extension,
                    status="duplicate",
                    warnings=[f"Same content as {seen_hashes[content_hash]}; only the first copy was processed."],
                )
            )
            continue
        seen_hashes[content_hash] = source

        try:
            loaded = load_source_file(path, source, config)
            row = register[source]
            methods = sorted({str(record["metadata"].get("extraction_method", "")) for record in loaded.records if record["metadata"].get("extraction_method")})
            document_types = sorted({str(record["metadata"].get("document_type", "")) for record in loaded.records if record["metadata"].get("document_type")})
            row["extraction_method"] = ";".join(methods)
            row["document_type"] = "mixed" if len(document_types) > 1 else (document_types[0] if document_types else row.get("document_type", ""))
            shared_metadata = register_metadata(row)
            for record in loaded.records:
                record["doc_id"] = row["doc_id"]
                record["metadata"].update(shared_metadata)
                record["metadata"]["content_hash"] = content_hash
                output.records.append(record)
            status = "processed" if loaded.records else "empty"
            output.files.append(
                FilePreparationStatus(
                    source=source,
                    extension=extension,
                    status=status,
                    record_count=len(loaded.records),
                    warnings=loaded.warnings,
                )
            )
        except Exception as exc:
            output.files.append(
                FilePreparationStatus(source=source, extension=extension, status="failed", error=str(exc))
            )
    save_document_register(list(register.values()), config.document_register_path)
    return output
