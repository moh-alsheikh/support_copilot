"""Create and maintain the corpus document register.

The register is your catalog of source files. Re-running preparation adds new files but
does not replace titles, topics, years, or repository notes you have already entered.
"""

import csv
from pathlib import Path
from typing import Any


REGISTER_FIELDS = [
    "doc_id",
    "title",
    "source_repository",
    "year",
    "document_type",
    "extraction_method",
    "topic",
    "path",
    "content_hash",
]


def infer_document_type(path: Path) -> str:
    """Return a readable initial type from a file extension."""

    extension = path.suffix.lower().lstrip(".")
    return {
        "md": "markdown",
        "markdown": "markdown",
        "txt": "text",
        "jpg": "scanned_image",
        "jpeg": "scanned_image",
        "png": "scanned_image",
        "tif": "scanned_image",
        "tiff": "scanned_image",
        "bmp": "scanned_image",
    }.get(extension, extension)


def load_document_register(path: str | Path) -> list[dict[str, str]]:
    """Read register rows, returning an empty list before the first preparation run."""

    register_path = Path(path)
    if not register_path.exists():
        return []
    with register_path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def save_document_register(rows: list[dict[str, str]], path: str | Path) -> Path:
    """Write register rows while retaining any extra columns you added."""

    output_path = Path(path)
    fieldnames = list(REGISTER_FIELDS)
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def ensure_document_register(
    source_files: list[Path],
    raw_dir: str | Path,
    register_path: str | Path,
    content_hashes: dict[str, str],
) -> dict[str, dict[str, str]]:
    """Add missing files to the register and preserve all existing manually entered fields."""

    raw_path = Path(raw_dir)
    output_path = Path(register_path)
    rows = load_document_register(output_path)
    by_path = {row.get("path", ""): row for row in rows}
    used_ids = {row.get("doc_id", "") for row in rows}

    for source_path in source_files:
        relative = source_path.relative_to(raw_path).as_posix()
        if relative not in by_path:
            next_number = len(rows) + 1
            doc_id = f"DOC_{next_number:04d}"
            while doc_id in used_ids:
                next_number += 1
                doc_id = f"DOC_{next_number:04d}"
            row = {
                "doc_id": doc_id,
                "title": source_path.stem.replace("_", " ").replace("-", " ").title(),
                "source_repository": "",
                "year": "",
                "document_type": infer_document_type(source_path),
                "extraction_method": "",
                "topic": "",
                "path": relative,
                "content_hash": content_hashes.get(relative, ""),
            }
            rows.append(row)
            by_path[relative] = row
            used_ids.add(doc_id)
        else:
            by_path[relative]["content_hash"] = content_hashes.get(relative, by_path[relative].get("content_hash", ""))

    save_document_register(rows, output_path)
    return by_path


def register_metadata(row: dict[str, str]) -> dict[str, Any]:
    """Select non-empty register fields that should travel with every record."""

    metadata: dict[str, Any] = {}
    for key in ("doc_id", "title", "source_repository", "year", "topic"):
        value = row.get(key, "").strip()
        if value:
            metadata[key] = value
    return metadata
