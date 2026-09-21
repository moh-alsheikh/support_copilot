"""Load TXT and Markdown files into blocks with line-range citations.

Blank lines provide natural boundaries in notes and Markdown. Keeping line ranges lets you open the
original file and verify a retrieved claim without relying only on a chunk number.
"""

from pathlib import Path

from .base import LoadResult, build_record


def read_text_file(path: str | Path) -> tuple[str, str]:
    """Read common text encodings and return `(text, encoding_used)`."""

    raw = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"Could not decode {path} as UTF-8 or Windows-1252 text.")


def split_line_blocks(text: str) -> list[tuple[str, int, int]]:
    """Split text on blank lines and retain one-based start/end line numbers."""

    blocks: list[tuple[str, int, int]] = []
    current: list[str] = []
    start_line = 1
    lines = text.splitlines()
    for line_number, line in enumerate(lines, start=1):
        if line.strip():
            if not current:
                start_line = line_number
            current.append(line.rstrip())
        elif current:
            blocks.append(("\n".join(current).strip(), start_line, line_number - 1))
            current = []
    if current:
        blocks.append(("\n".join(current).strip(), start_line, len(lines)))
    return blocks


def load_text(path: str | Path, source: str) -> LoadResult:
    """Load a text or Markdown file as line-addressable records."""

    text, encoding = read_text_file(path)
    document_type = "markdown" if Path(path).suffix.lower() in {".md", ".markdown"} else "text"
    records = [
        build_record(
            block,
            source=source,
            location=f"lines:{start}-{end}",
            document_type=document_type,
            extraction_method="text_parser",
            extra_metadata={"encoding": encoding},
        )
        for block, start, end in split_line_blocks(text)
        if block
    ]
    warnings = [] if records else ["The file contains no readable text."]
    return LoadResult(records=records, warnings=warnings)
