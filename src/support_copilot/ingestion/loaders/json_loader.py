"""Load JSON and JSON Lines without indexing punctuation-heavy source syntax.

Nested values become readable dotted keys such as `engine.temperature: 120`. A top-level JSON list
creates one record per item; JSON Lines creates one record per non-empty line.
"""

import json
from pathlib import Path
from typing import Any

from .base import LoadResult, build_record
from .text_loader import read_text_file


def flatten_json(value: Any, prefix: str = "") -> list[str]:
    """Recursively convert JSON leaves into readable `path: value` lines."""

    if isinstance(value, dict):
        lines: list[str] = []
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            lines.extend(flatten_json(child, child_prefix))
        return lines
    if isinstance(value, list):
        lines = []
        for index, child in enumerate(value):
            lines.extend(flatten_json(child, f"{prefix}[{index}]" if prefix else f"[{index}]"))
        return lines

    label = prefix or "value"
    if value is None:
        display = "null"
    elif isinstance(value, bool):
        display = str(value).lower()
    else:
        display = str(value)
    return [f"{label}: {display}"]


def _record_for_json(value: Any, source: str, location: str, method: str) -> dict[str, Any] | None:
    """Create one searchable normalized record from a parsed JSON value."""

    text = "\n".join(flatten_json(value)).strip()
    if not text:
        return None
    return build_record(
        text,
        source=source,
        location=location,
        document_type="jsonl" if method == "jsonl_parser" else "json",
        extraction_method=method,
    )


def load_json(path: str | Path, source: str) -> LoadResult:
    """Load `.json` or `.jsonl` into structured text records."""

    text, _ = read_text_file(path)
    records: list[dict[str, Any]] = []
    if Path(path).suffix.lower() == ".jsonl":
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc.msg}") from exc
            record = _record_for_json(value, source, f"line:{line_number}", "jsonl_parser")
            if record:
                records.append(record)
    else:
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc
        items = value if isinstance(value, list) else [value]
        for index, item in enumerate(items):
            location = f"$[{index}]" if isinstance(value, list) else "$"
            record = _record_for_json(item, source, location, "json_parser")
            if record:
                records.append(record)

    warnings = [] if records else ["The JSON file contains no searchable scalar values."]
    return LoadResult(records=records, warnings=warnings)
