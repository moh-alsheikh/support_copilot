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

`data/raw/` ships with a small, genuinely support-flavored sample knowledge base for a fictional product
("Lumen Sync," a desktop backup/sync client), so the pipeline is runnable and testable out of the
box: `lumen_user_guide.md` (product manual), `support_faq.md` (FAQ), and `support_policy.txt`
(refund/SLA policy). It is intentionally small — enough to exercise multi-file discovery, distinct
`lines:A-B` citations across three separate sources, and the full grounded-answer/abstention
contract, not to demonstrate scale. Swap in your own product manuals, support policies, or KB
articles under `data/raw/` for a real deployment; the pipeline treats every source the same way
regardless of size or domain. Expected behavior when preparing it:

- 3 files, a low double-digit number of line-addressable records, no duplicates;
- OCR never triggers — nothing here is a scanned image or a text-less PDF;
- `evaluation/datasets/evaluation_questions.json` has matching `(source, location)` ground truth
  for this knowledge base; update it if you replace or edit the sample files.

## Complete workflow

```bash
cp .env.example .env

# 1. Inspect optional OCR dependencies.
python prepare_corpus.py --check-dependencies

# 2. Extract files and write normalized records.
python prepare_corpus.py

# 3. Rebuild Chroma and BM25 from normalized records.
python scripts/ingest_documents.py
# Repeated automated build after reviewing generated state:
python scripts/ingest_documents.py --force

# 4. Inspect ranking without generation.
python scripts/inspect_index.py --mode semantic
python scripts/inspect_index.py --mode bm25
python scripts/inspect_index.py --mode hybrid

# 5. Measure retrieval before evaluating answers.
python scripts/run_evaluation.py retrieval --mode hybrid --top-k 4 --run-name baseline
python scripts/run_evaluation.py answers --mode hybrid --top-k 4 --run-name baseline

# 6. Open the workbench.
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

Indexing does a full rebuild of Chroma and BM25 on every run — deliberately simple, not
incremental (see `docs/THE_JOURNEY_TO_PRODUCTION.md` for what changes at real scale). Repeating
the command produces the same collection size, removes stale records, and never duplicates nodes.
When indexes already exist, the script asks for confirmation; `--force` is the explicit automation
path.

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
- **Readiness & Limitations** — an editable production-readiness checklist grounded in this
  project's own documented gaps (see `docs/THE_JOURNEY_TO_PRODUCTION.md`).

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
- Existing index replacement: review it, then confirm, or use `--force` for an intentional rebuild.
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
