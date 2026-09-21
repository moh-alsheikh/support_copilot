"""Load modern Microsoft Word `.docx` files.

Word files do not expose stable page numbers because pagination depends on fonts and printers.
This loader therefore cites paragraph numbers and table/row coordinates from the document model.
"""

from pathlib import Path

from .base import LoadResult, build_record


def load_docx(path: str | Path, source: str) -> LoadResult:
    """Extract non-empty Word paragraphs and table rows as separate records."""

    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError("Install python-docx with: pip install -r requirements.txt") from exc

    document = Document(str(path))
    records = []
    for paragraph_number, paragraph in enumerate(document.paragraphs, start=1):
        text = paragraph.text.strip()
        if text:
            records.append(
                build_record(
                    text,
                    source=source,
                    location=f"paragraph:{paragraph_number}",
                    document_type="docx",
                    extraction_method="python_docx",
                    extra_metadata={"paragraph_style": paragraph.style.name if paragraph.style else ""},
                )
            )

    for table_number, table in enumerate(document.tables, start=1):
        header = [cell.text.strip() for cell in table.rows[0].cells] if table.rows else []
        for row_number, row in enumerate(table.rows, start=1):
            cells = [cell.text.strip() for cell in row.cells]
            if not any(cells):
                continue
            if row_number > 1 and any(header):
                parts = [f"{name or f'column_{index}'}: {value}" for index, (name, value) in enumerate(zip(header, cells), start=1) if value]
            else:
                parts = [value for value in cells if value]
            records.append(
                build_record(
                    "\n".join(parts),
                    source=source,
                    location=f"table:{table_number}/row:{row_number}",
                    document_type="docx",
                    extraction_method="python_docx",
                )
            )

    warnings = [] if records else ["The Word file contains no readable paragraphs or table rows."]
    return LoadResult(records=records, warnings=warnings)
