"""Load CSV files as one searchable record per data row.

Column names are repeated in each record (`temperature: 120`) so a retrieved row keeps its meaning
after it has been separated from the rest of the table.
"""

import csv
from pathlib import Path

from .base import LoadResult, build_record
from .text_loader import read_text_file


def load_csv(path: str | Path, source: str) -> LoadResult:
    """Parse a CSV file and preserve the original one-based file row number."""

    text, encoding = read_text_file(path)
    reader = csv.DictReader(text.splitlines())
    if not reader.fieldnames:
        return LoadResult(warnings=["The CSV file has no header row."])

    records = []
    for row_number, row in enumerate(reader, start=2):
        values = [f"{key}: {value}" for key, value in row.items() if key and value not in (None, "")]
        if values:
            records.append(
                build_record(
                    "\n".join(values),
                    source=source,
                    location=f"row:{row_number}",
                    document_type="csv",
                    extraction_method="csv_parser",
                    extra_metadata={"encoding": encoding},
                )
            )
    warnings = [] if records else ["The CSV file has headers but no non-empty data rows."]
    return LoadResult(records=records, warnings=warnings)
