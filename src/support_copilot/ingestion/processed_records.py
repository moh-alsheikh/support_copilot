"""Persist normalized source records as newline-delimited JSON.

JSONL keeps one record per line, is easy to inspect in an editor, and lets indexing reuse prepared
text without reopening every source file or rerunning OCR.
"""

import json
from pathlib import Path
from typing import Any


def write_records(records: list[dict[str, Any]], path: str | Path) -> Path:
    """Write records as UTF-8 JSONL and return the output path."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Read and minimally validate normalized JSONL records."""

    input_path = Path(path)
    records: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record.get("text"), str) or not isinstance(record.get("metadata"), dict):
                raise ValueError(f"Invalid processed record on line {line_number}.")
            records.append(record)
    return records
