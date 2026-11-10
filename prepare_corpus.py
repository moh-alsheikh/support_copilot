"""Extract every supported source file into reusable normalized JSONL records.

Raw files are never modified. One failed or unsupported file is reported without stopping healthy
files. Run this command before indexing whenever the source corpus changes.

Usage:
    python prepare_corpus.py --check-dependencies
    python prepare_corpus.py
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from support_copilot.config import AppConfig
from support_copilot.ingestion.document_loader import SUPPORTED_EXTENSIONS, prepare_documents
from support_copilot.ingestion.ocr import check_ocr_dependencies
from support_copilot.ingestion.processed_records import write_records


def main() -> None:
    """Inspect optional OCR tools or prepare the complete recursive corpus."""

    parser = argparse.ArgumentParser(description="Prepare multi-format files for LlamaIndex.")
    parser.add_argument("--check-dependencies", action="store_true")
    args = parser.parse_args()
    config = AppConfig.from_env()

    if args.check_dependencies:
        print("Supported extensions:", ", ".join(sorted(SUPPORTED_EXTENSIONS)))
        for dependency, status in check_ocr_dependencies().items():
            print(f"{dependency}: {status}")
        print("Missing or broken OCR tools affect only images and scanned PDF pages. See OCR_SETUP.md.")
        return

    config.raw_data_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_documents(config)
    write_records(prepared.records, config.processed_records_path)
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = config.reports_dir / "corpus_preparation.json"
    report_path.write_text(
        json.dumps(
            {
                "summary": prepared.summary(),
                "record_count": len(prepared.records),
                "files": [item.__dict__ for item in prepared.files],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Files discovered: {len(prepared.files)}")
    print(f"Normalized records: {len(prepared.records)}")
    for status, count in sorted(prepared.summary().items()):
        print(f"{status}: {count}")
    for item in prepared.files:
        for warning in item.warnings:
            print(f"WARNING {item.source}: {warning}")
        if item.error:
            print(f"ERROR {item.source}: {item.error}")
    print(f"Records: {config.processed_records_path}")
    print(f"Register: {config.document_register_path}")


if __name__ == "__main__":
    main()
