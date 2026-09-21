# Multi-Format Ingestion

## Folder Contract

Original documents belong under `data/raw/`. Discovery is recursive, so subfolders can represent
repositories, teams, topics, or source systems. Generated normalized records and indexes never
replace originals.

## Extraction Strategy

| Format | Extraction | Location |
| --- | --- | --- |
| PDF | pypdf, then page OCR fallback | `page:N` |
| Image | Tesseract | `image:1` |
| TXT/Markdown | decoded line blocks | `lines:A-B` |
| DOCX | paragraphs and table rows | `paragraph:N`, `table:T/row:R` |
| CSV | header-labelled rows | `row:N` |
| JSON | structurally flattened records | `$`, `$[N]` |
| JSONL | one parsed line per record | `line:N` |

JSON is parsed structurally. Nested keys become readable dotted labels instead of punctuation-heavy
raw JSON. Invalid JSON reports its line/column and does not stop healthy files from preparing.

## OCR Is Optional

Text PDFs use their existing text layer. OCR is attempted only for pages with too little extracted
text. Tesseract handles recognition; Poppler renders PDF pages. Missing OCR tools produce warnings
while TXT, Markdown, DOCX, CSV, JSON, and text PDFs continue to work.

## Duplicate Detection

Preparation hashes file bytes with SHA-256. When two paths contain identical bytes, the first is
prepared and the later path is reported as a duplicate. Different files sharing the same basename
remain distinct because metadata stores the relative path under `data/raw/`.

## Normalized Record Contract

```python
{
    "doc_id": "DOC_0004",
    "text": "engine.temperature: 120",
    "metadata": {
        "source": "structured/tests.json",
        "location": "$[4]",
        "document_type": "json",
        "extraction_method": "json_parser",
    },
}
```

The LlamaIndex adapter turns each record into a `Document`. Chunk nodes inherit metadata, so every
retrieval result can be traced to the original location.
