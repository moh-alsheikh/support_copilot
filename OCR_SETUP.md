# OCR Setup

OCR is optional. TXT, Markdown, DOCX, CSV, JSON, JSONL, and text-layer PDFs work without Tesseract or
Poppler. Images require Tesseract. Scanned PDF pages require both Tesseract and Poppler.

## macOS

```bash
brew install tesseract poppler
python prepare_corpus.py --check-dependencies
```

## Ubuntu Or Debian

```bash
sudo apt-get update
sudo apt-get install tesseract-ocr poppler-utils
python prepare_corpus.py --check-dependencies
```

## Windows

Install Tesseract OCR and Poppler, add both executable directories to `PATH`, restart the terminal,
and run the dependency check. Installation locations vary; follow the official project packages
approved by your organization.

## How Fallback Works

PDF pages first use `pypdf`. OCR runs only when extracted text is shorter than `OCR_MIN_CHARS` and
`OCR_ENABLED=true`. A useful text layer is retained when OCR produces less text. Failures are
reported per page and do not stop unrelated files.

OCR text can contain recognition errors. Verify critical values against the original scan and keep
the `extraction_method` metadata visible during corpus review.
