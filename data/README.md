# Corpus Folders

Put every original source document under `data/raw/`. Subfolders are encouraged and discovered
recursively. Do not place originals in `processed/`, `metadata/`, `storage/`, or `evaluation/`.

```text
data/
|-- raw/            original source files you supply
|-- processed/      generated normalized JSONL records
`-- metadata/       generated document register and manually maintained metadata
```

Supported source extensions are PDF, PNG, JPG, JPEG, TIF, TIFF, BMP, TXT, Markdown, DOCX, CSV,
JSON, and JSONL. Legacy `.doc` files must be exported as `.docx`.

Each prepared record keeps a universal location: PDF page, image number, text line range, Word
paragraph/table row, CSV row, JSON path, or JSONL line. Relative paths under `raw/` remain in the
`source` field, so `reports/manual.pdf` and `archive/manual.pdf` do not collide.

Run `python prepare_corpus.py`, inspect its warnings, and then run
`python scripts/ingest_documents.py`. Preparation never changes the raw files.

Inspect `metadata/document_register.csv` after preparation. Add project titles, topics, years, and
repository notes there; later preparation runs preserve your manually maintained columns.
