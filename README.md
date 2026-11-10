# Support Copilot

A local-first, citation-grounded RAG assistant: ask a question about a product manual, support
policy, or other technical documentation in natural language, and get an answer that either cites
the exact source and location it came from, or honestly says it doesn't know.

It is not "chat with your files." A complete run produces inspectable ingestion, persistent
retrieval indexes (semantic + keyword), grounded answers with verifiable citations, quality
reports, and a Streamlit workbench to drive all of it.

```text
Raw multi-format files
  -> extraction/OCR -> normalized records with source + location
  -> LlamaIndex Documents -> Nodes
  -> Chroma semantic index + persisted BM25 index
  -> semantic/BM25/hybrid retrieval -> RRF -> transparent rerank
  -> numbered evidence -> QA/refine contracts -> normalized answer -> universal citations
  -> retrieval reports + answer reports + human rubric
```

## Why this exists

Most "RAG demos" stop at "it answers questions." This project's actual subject is the harder,
less demoed half: **making the answer verifiable and making refusal a first-class outcome.**
Concretely:

- Every claim in an answer is tied to a `[Source N]` marker, and that marker resolves to real
  document metadata — never to something the model typed.
- If the knowledge base doesn't support an answer, the system returns one fixed, exact message instead of
  guessing. That contract is enforced in code, not left to a prompt (see `docs/known-findings.md`
  for the real failure that made this necessary).
- Every document format — PDF, scanned image, DOCX, CSV, JSON — is normalized to a **universal
  location** (`page:7`, `table:2/row:4`, `row:15`, `$[4]`) so a citation always means something
  concrete, regardless of source format.

## Where files go

Put source documents under:

```text
data/raw/
```

Subfolders are supported and scanned recursively. Don't put originals in `data/processed/`,
`data/metadata/`, `storage/`, or `evaluation/reports/` — those are generated.

Supported extensions: `.pdf`, `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`, `.txt`, `.md`,
`.markdown`, `.docx`, `.csv`, `.json`, and `.jsonl`. Convert legacy `.doc` and Google Docs files to
a supported format first.

### The sample knowledge base

`data/raw/` ships 11 files centred on one product, the **Raspberry Pi 5**, plus related support
and safety data, so the pipeline is runnable and testable out of the box and every supported format
is exercised:

| Format | Files |
| --- | --- |
| PDF (text layer) | Pi 5 product brief, 27W USB-C power supply brief, Active Cooler brief, Pi 5 safety and user guide |
| PDF (scanned, OCR only) | `cpsc-battery-charger-safety-meeting-1998-scan.pdf` — 3 image-only pages |
| Markdown (AsciiDoc tables) | `led-warning-flash-codes.md`, `power-supplies.md` |
| Images (OCR) | `bootloader-diagnostics.png`, `gpio-pinout-diagram.png` |
| JSON | `cpsc-power-bank-recalls.json` — 32 US CPSC power-bank recalls |
| CSV | `support-tickets-sample.csv` — 135 rows of generic customer-support Q&A |

Sources and licenses for every file are in [`data/SOURCES.md`](data/SOURCES.md). Several
questions need more than one file (for example, LED codes plus power-supply requirements). Swap
in your own product manuals, support policies, or KB articles under `data/raw/`; the pipeline
treats every source the same way. Expected behavior when preparing it:

- 11 files, about 320 normalized records; OCR runs on the two images and the scanned PDF (needs
  Tesseract and Poppler — see [`OCR_SETUP.md`](OCR_SETUP.md); without them those files are
  skipped with a warning);
- table rows in the Markdown files become one labelled record each (`Long flashes: 0 | Short
  flashes: 7 | Status: …`), keeping their original `lines:A-B` location;
- `evaluation/datasets/evaluation_questions.json` has 23 questions (19 answerable, 4 deliberately
  unanswerable near-misses) with `(source, location)` ground truth verified against these files;
  update it if you replace or edit them.

## Complete workflow

Requires Python 3.11 or 3.12 (3.12 is what's used day to day) and [Ollama](https://ollama.com) with
`llama3.2:3b` and `nomic-embed-text` pulled. Python 3.13+ doesn't work on Intel Macs: `chromadb`
depends on `onnxruntime`, which publishes no Intel-Mac builds for those versions.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# 1. Inspect optional OCR dependencies.
python prepare_corpus.py --check-dependencies

# 2. Extract files and write normalized records.
python prepare_corpus.py

# 3. Update Chroma and BM25 from normalized records. Only new or changed records are embedded;
#    an unchanged corpus finishes in seconds. The first run builds everything (~4 min on a CPU).
python scripts/ingest_documents.py
# Discard the indexes and rebuild from scratch (--force skips the confirmation prompt):
python scripts/ingest_documents.py --full-rebuild

# 4. Inspect ranking without generation. Without --question, the default is
#    "Which power supply is recommended for the Raspberry Pi 5?".
python scripts/inspect_index.py --mode semantic
python scripts/inspect_index.py --mode bm25
python scripts/inspect_index.py --mode hybrid

# Ask your own question. Each example below is answered by a different file type:
python scripts/inspect_index.py --mode hybrid --question "What is the maximum fan speed of the Raspberry Pi Active Cooler?"              # PDF
python scripts/inspect_index.py --mode hybrid --question "Below what supply voltage does the Raspberry Pi low-voltage detection trigger?"   # Markdown
python scripts/inspect_index.py --mode hybrid --question "Why were IKEA VARMFRONT power banks recalled?"                                    # JSON
python scripts/inspect_index.py --mode hybrid --question "In what situations can I request a refund?"                                      # CSV
python scripts/inspect_index.py --mode hybrid --question "What did the bootloader diagnostics screen report about the SD card?"            # image (OCR)
python scripts/inspect_index.py --mode hybrid --question "Which GPIO pin carries the UART TXD signal on the 40-pin header?"                # image (OCR)
python scripts/inspect_index.py --mode hybrid --question "What are the four parts of a battery described in the CPSC battery charger safety meeting?"   # scanned PDF (OCR)

# The pinout diagram is the hardest OCR source: the two pin columns come out interleaved, so labels
# like "GPIO 14 (TXD)" survive but pin positions don't. It ranks 3rd, behind two GPIO-related text
# chunks; see OCR_SETUP.md, "OCR Quality".

# Not answerable from the knowledge base: retrieval still returns the closest chunks (here, the
# 27W supply). Only the answer step refuses, which is what the near-miss evaluation questions test.
python scripts/inspect_index.py --mode hybrid --question "What is the maximum output of the Raspberry Pi 45W USB-C Power Supply?"

# Widen the result list (default comes from TOP_K in .env). Here the answer (product brief,
# page 3) ranks 6th, so it only shows up beyond the default top 4:
python scripts/inspect_index.py --mode hybrid --top-k 8 --question "What is the operating temperature range of the Raspberry Pi 5?"

# 5. Ask with full generation: retrieval + a grounded, cited answer from the LLM.
#    On a CPU-only machine expect about a minute per answer (longer for the first, while the model loads).
python scripts/ask.py --question "Which power supply is recommended for the Raspberry Pi 5?"
python scripts/ask.py --question "What is the maximum fan speed of the Raspberry Pi Active Cooler?"
python scripts/ask.py --question "Below what supply voltage does the Raspberry Pi low-voltage detection trigger?"
python scripts/ask.py --question "Why were IKEA VARMFRONT power banks recalled?"
python scripts/ask.py --question "In what situations can I request a refund?"

# Near-miss questions: the knowledge base covers a similar product, not this one. The answer
# should be the exact refusal, even though related chunks are retrieved.
python scripts/ask.py --question "How fast does the fan in the official Raspberry Pi 5 Case spin?"
python scripts/ask.py --question "What is the maximum output of the Raspberry Pi 45W USB-C Power Supply?"

# Scanned PDF: every page of this 1998 CPSC meeting log is OCR'd (needs Tesseract + Poppler, see
# OCR_SETUP.md). The second question is a near-miss: page 3 mentions civil penalties, but for a
# different meeting about fabrics, so the answer should be the refusal.
python scripts/ask.py --question "According to the 1998 CPSC meeting on battery charger safety, what can happen when lead-acid batteries are overcharged?"
python scripts/ask.py --question "What civil penalty amounts did the CPSC set for unsafe battery chargers?"

# A known small-model limitation: the answer ("no models support USB-PPS") is retrieved, but
# llama3.2:3b usually refuses anyway. See docs/learning/local-llm-model-selection.md.
python scripts/ask.py --question "Do any Raspberry Pi models support USB-PPS?"

# Try a different Ollama model for one run without editing .env (after `ollama pull phi4-mini`):
OLLAMA_LLM_MODEL=phi4-mini python scripts/ask.py --question "Do any Raspberry Pi models support USB-PPS?"

# 6. Measure retrieval before evaluating answers.
python scripts/run_evaluation.py retrieval --mode hybrid --top-k 4 --run-name baseline
python scripts/run_evaluation.py answers --mode hybrid --top-k 4 --run-name baseline

# 7. Open the workbench.
streamlit run app.py
```

## Preparation contract

Raw files remain unchanged. Preparation creates `data/processed/document_records.jsonl`, updates
`data/metadata/document_register.csv` without erasing prior metadata, detects duplicate content,
and writes `evaluation/reports/corpus_preparation.json`.

Universal locations make evidence verifiable regardless of source format:

| Source | Example location |
| --- | --- |
| PDF | `page:7` |
| Image | `image:1` |
| Text/Markdown | `lines:20-35` |
| DOCX paragraph | `paragraph:12` |
| DOCX table | `table:2/row:4` |
| CSV | `row:15` |
| JSON | `$[4]` |
| JSONL | `line:12` |

## Indexing behavior

Indexing is incremental. LlamaIndex's `IngestionPipeline` keeps a document store of per-record
hashes (`storage/ingestion_pipeline/`), so each run splits and embeds only new or changed records
and deletes the vectors of records that disappeared (`UPSERTS_AND_DELETE`). On the sample
knowledge base, an unchanged re-run takes about 5 seconds instead of about 4.5 minutes. BM25 has no
incremental mode, so it's rebuilt from all current nodes each run, which is cheap. The script
reports what it did:

```text
Mode: incremental
Documents: 323 (added 0, changed 0, removed 0, unchanged 323)
Nodes embedded this run: 0
```

A full rebuild happens automatically when there is nothing to diff against (no index or document
store yet) or when a setting that invalidates every vector changes (chunk size or overlap,
embedding model, distance metric), and on request with `--full-rebuild` (asks to confirm) or
`--force` (no prompt). Chunk IDs are deterministic, so rebuilding an unchanged corpus reproduces
the same IDs and corpus fingerprint.

Every successful build writes `storage/index_manifest.json`: node count, corpus SHA-256
fingerprint, chunk policy, embedding model, and BM25 policy version. Query-time loading rejects a
missing or incompatible manifest rather than silently searching stale term statistics.

Chroma's vector similarity is computed as cosine distance (`hnsw:space="cosine"`, set explicitly in
`indexing/vector_store.py`), not Chroma's own `"l2"` default — chosen so ranking doesn't silently
depend on the embedding model happening to return normalized vectors. See
[`docs/learning/vector-similarity-scoring.md`](docs/learning/vector-similarity-scoring.md) for the
full reasoning and verification.

## Grounding and citation contract

- semantic and real BM25 evidence are fused before generation;
- candidate K is `max(50, top_k * 5)` and final top-k is a separate decision — see
  `docs/known-findings.md` for why that constant exists;
- fused evidence retains semantic and BM25 component ranks for the UI and error analysis;
- no evidence returns the exact abstention message without calling a provider;
- QA and refine prompts both forbid outside knowledge and invented facts;
- each context block has a stable `[Source N]` label, source, universal location, and chunk ID;
- citation rows come from retrieved Node metadata, never from model output;
- citation numbers stay tied to evidence positions, even for duplicate locations;
- a response beginning with the abstention message and then adding claims is normalized back to
  the exact message.

Read [Grounding, Abstention, And Citations](docs/grounding-abstention-citations.md) before
changing the prompts or synthesizer.

## Evaluation contract

Expected sources and locations align as exact pairs. `manual.pdf` at `page:2` does not satisfy an
expectation for `manual.pdf` at `page:7`. Retrieval reports compute Hit@K, Recall@K, context
precision, reciprocal rank, and MRR over supported questions, and report unsupported questions
separately.

Answer reports record evidence, universal citations, latency, provider, model, retrieval mode,
citation marker validity, false answers, false abstentions, and whether expected evidence was
cited. These deterministic checks are a smoke signal, not a substitute for the human 0/1/2 rubric.

## Package reading path

| Stage | Start here | What to trace |
| --- | --- | --- |
| preparation | `prepare_corpus.py`, `ingestion/pipeline.py` | one raw file to normalized records |
| framework adapter | `ingestion/document_adapter.py`, `ingestion/parsers.py` | record to Document to Node |
| indexing | `indexing/index_manager.py` | same Nodes into Chroma and BM25 plus manifest |
| retrieval | `retrieval/retriever.py` | candidate K, branch results, RRF, rerank |
| generation | `generation/prompts.py`, `generation/synthesizer.py` | `[Source N]` into normalized answer |
| citations | `generation/citations.py` | evidence position to source/location row |
| evaluation | `evaluation/runner.py`, `evaluation/evaluators.py` | expected pair to metric and failure row |
| interfaces | `services/`, `app.py`, `scripts/` | one service reused by CLI, UI, and reports |

## Failure boundaries

Diagnose in pipeline order:

1. **Preparation:** unsupported, empty, invalid, duplicate, OCR, or parser warning.
2. **Node creation:** missing metadata, unstable IDs, harmful chunk boundary.
3. **Index lifecycle:** stale/missing manifest, empty Chroma, missing BM25.
4. **Retrieval:** expected pair absent or too low in semantic/BM25/hybrid ranks.
5. **Grounding:** evidence present but answer adds unsupported content or fails to cite.
6. **Evaluation:** incorrect ground truth, misleading average, incomplete human rubric.

Changing the prompt is appropriate only after the necessary evidence is confirmed present.

## Streamlit views

- **Ask Assistant** — provider, retriever, top-k, answer, citations, and retrieved evidence.
- **Knowledge Base Health** — raw files, supported formats, preparation warnings, normalized previews, and
  the document register.
- **Evaluation Dashboard** — named retrieval/answer runs, metric chart, reports, and rubric
  downloads.

Ollama is the local default (fully offline). Anthropic is an explicit, opt-in online comparison
provider and requires `LLM_PROVIDER=anthropic` plus `ANTHROPIC_API_KEY`.

## Tests

```bash
python -m pytest -q
```

Covers recursive discovery, duplicate files, TXT line locations, nested JSON, DOCX paragraphs and
tables, universal citations, evidence-label alignment, prompt/refine parity, contradictory
abstention, idempotent Chroma builds, index manifests, RRF candidate policy, and pair-aware
metrics. No test requires internet, Ollama, Tesseract, or the sample knowledge base.

## Troubleshooting

- Missing processed JSONL: run `python prepare_corpus.py`.
- Empty/missing index: run `python scripts/ingest_documents.py`.
- Index looks stale or inconsistent: run `python scripts/ingest_documents.py --full-rebuild`.
- Missing/incompatible manifest: rebuild both indexes; don't copy only one generated directory.
- OCR dependency warning: text formats still work; see `OCR_SETUP.md` for scanned sources.
- Ollama connection failure: start Ollama and verify both models with `ollama list`.
- Wrong citation: inspect the prepared record's location before touching generation prompts.

## Design notes

- [`docs/ARCHITECTURE_ASSESSMENT.md`](docs/ARCHITECTURE_ASSESSMENT.md) — an honest, scored
  self-assessment across architecture, code-level design, and production-readiness.
- [`docs/THE_JOURNEY_TO_PRODUCTION.md`](docs/THE_JOURNEY_TO_PRODUCTION.md) — stage-by-stage plan
  for what changes (and what doesn't) between this scale and a real production deployment.
- [`docs/known-findings.md`](docs/known-findings.md) — real bugs found and fixed during
  development, with symptom, diagnosis, fix, and lesson.
- [`docs/DESIGN_QUESTIONS.md`](docs/DESIGN_QUESTIONS.md) — open architecture questions worth
  reasoning through before extending this project.
- [`docs/BACKLOG.md`](docs/BACKLOG.md) — concrete features and improvements not yet built.
