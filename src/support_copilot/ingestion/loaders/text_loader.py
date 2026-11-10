"""Load TXT and Markdown files into blocks with line-range citations.

Blank lines provide natural boundaries in notes and Markdown. Keeping line ranges lets you open the
original file and verify a retrieved claim without relying only on a chunk number. Tables with a
header row are the exception: each row becomes one record labelled with its column names and the
table title or nearest heading, so a row like `| 0 | 7 | Kernel image not found` stays findable.
"""

from dataclasses import dataclass, field
from pathlib import Path
import re

from .base import LoadResult, build_record

HEADING = re.compile(r"^(?:=+|#+)\s+(\S.*)$")
MARKDOWN_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?\s*$")
DECLARED_COLUMNS = re.compile(r'cols="([^"]+)"')
HEADER_OPTION = re.compile(r'options="[^"]*header|%header')


@dataclass
class _Table:
    """Zero-based `start`/`end` cover the whole table, including its title and attribute lines."""

    start: int
    end: int
    rows: list[tuple[str, int, int]] = field(default_factory=list)


def read_text_file(path: str | Path) -> tuple[str, str]:
    """Read common text encodings and return `(text, encoding_used)`."""

    raw = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"Could not decode {path} as UTF-8 or Windows-1252 text.")


def _line_blocks(lines: list[str], skip: set[int]) -> list[tuple[str, int, int]]:
    """Split on blank lines (and skipped table lines), keeping one-based start/end line numbers."""

    blocks: list[tuple[str, int, int]] = []
    current: list[str] = []
    start_line = 1
    for index, line in enumerate(lines):
        if line.strip() and index not in skip:
            if not current:
                start_line = index + 1
            current.append(line.rstrip())
        elif current:
            blocks.append(("\n".join(current).strip(), start_line, index))
            current = []
    if current:
        blocks.append(("\n".join(current).strip(), start_line, len(lines)))
    return blocks


def split_line_blocks(text: str) -> list[tuple[str, int, int]]:
    """Split text on blank lines and retain one-based start/end line numbers."""

    return _line_blocks(text.splitlines(), set())


def _row_text(context: str | None, header: list[str], values: list[str]) -> str:
    labelled = " | ".join(f"{name}: {value}" for name, value in zip(header, values) if value)
    return f"{context} — {labelled}" if context else labelled


def _declared_columns(attributes: str) -> int | None:
    match = DECLARED_COLUMNS.search(attributes)
    if not match:
        return None
    total = 0
    for spec in match.group(1).split(","):
        repeat = re.match(r"\s*(\d+)\*", spec)
        total += int(repeat.group(1)) if repeat else 1
    return total


def _closing_delimiter(lines: list[str], open_index: int) -> int | None:
    return next((i for i in range(open_index + 1, len(lines)) if lines[i].strip() == "|==="), None)


def _asciidoc_table(lines: list[str], open_index: int, heading: str | None) -> _Table | None:
    """Parse a `|===` table with a header row; return None to keep plain blank-line behavior."""

    close = _closing_delimiter(lines, open_index)
    if close is None:
        return None

    start, title, attributes = open_index, None, ""
    while start > 0:
        above = lines[start - 1].strip()
        if above.startswith("[["):
            pass
        elif above.startswith("["):
            attributes += above
        elif above.startswith(".") and not above.startswith(".."):
            title = above[1:].strip()
        else:
            break
        start -= 1

    cells: list[tuple[str, int]] = []
    first_line_cells: list[str] | None = None
    implicit_header = False
    for index in range(open_index + 1, close):
        line = lines[index].strip()
        if not line:
            continue
        if not line.startswith("|"):
            return None
        line_cells = [cell.strip() for cell in line.split("|")[1:]]
        if first_line_cells is None:
            first_line_cells = line_cells
            implicit_header = len(line_cells) > 1 and index + 1 < close and not lines[index + 1].strip()
        cells.extend((cell, index + 1) for cell in line_cells)

    if first_line_cells is None or not (HEADER_OPTION.search(attributes) or implicit_header):
        return None
    columns = _declared_columns(attributes) or len(first_line_cells)
    if columns < 2 or len(cells) % columns or len(cells) == columns:
        return None

    header = [cell for cell, _ in cells[:columns]]
    context = title or heading
    table = _Table(start=start, end=close)
    for offset in range(columns, len(cells), columns):
        row = cells[offset : offset + columns]
        table.rows.append((_row_text(context, header, [cell for cell, _ in row]), row[0][1], row[-1][1]))
    return table


def _markdown_cells(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _markdown_table(lines: list[str], header_index: int, heading: str | None) -> _Table | None:
    """Parse a Markdown pipe table; return None to keep plain blank-line behavior."""

    header = _markdown_cells(lines[header_index])
    if len(header) < 2:
        return None
    table = _Table(start=header_index, end=header_index + 1)
    index = header_index + 2
    while index < len(lines) and lines[index].strip() and "|" in lines[index]:
        values = _markdown_cells(lines[index])
        if len(values) != len(header):
            return None
        table.rows.append((_row_text(heading, header, values), index + 1, index + 1))
        table.end = index
        index += 1
    return table if table.rows else None


def _find_tables(lines: list[str]) -> list[_Table]:
    """Locate header-row tables in AsciiDoc (`|===`) or Markdown pipe syntax."""

    tables: list[_Table] = []
    heading: str | None = None
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        match = HEADING.match(stripped)
        if match:
            heading = match.group(1).strip()
        table = None
        if stripped == "|===":
            table = _asciidoc_table(lines, index, heading)
            if table is None:
                close = _closing_delimiter(lines, index)
                index = close + 1 if close is not None else index + 1
                continue
        elif (
            "|" in stripped
            and index + 1 < len(lines)
            and "|" in lines[index + 1]
            and MARKDOWN_SEPARATOR.match(lines[index + 1])
        ):
            table = _markdown_table(lines, index, heading)
        if table is not None:
            tables.append(table)
            index = table.end + 1
        else:
            index += 1
    return tables


def load_text(path: str | Path, source: str) -> LoadResult:
    """Load a text or Markdown file as line-addressable records."""

    text, encoding = read_text_file(path)
    document_type = "markdown" if Path(path).suffix.lower() in {".md", ".markdown"} else "text"
    lines = text.splitlines()
    tables = _find_tables(lines)
    table_lines = {index for table in tables for index in range(table.start, table.end + 1)}
    blocks = _line_blocks(lines, table_lines) + [row for table in tables for row in table.rows]
    records = [
        build_record(
            block,
            source=source,
            location=f"lines:{start}-{end}",
            document_type=document_type,
            extraction_method="text_parser",
            extra_metadata={"encoding": encoding},
        )
        for block, start, end in sorted(blocks, key=lambda item: item[1])
        if block
    ]
    warnings = [] if records else ["The file contains no readable text."]
    return LoadResult(records=records, warnings=warnings)
