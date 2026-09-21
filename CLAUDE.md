# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Support Copilot: a local-first, citation-grounded RAG assistant built on LlamaIndex. Ask a question
about a product manual, support policy, or other technical documentation; get an answer that cites
the exact source and location it came from, or an honest "I don't know."

The project's actual subject is making RAG answers verifiable, not just producing answers:
- Every claim in an answer must resolve to a `[Source N]` marker backed by real retrieved Node
  metadata — never text the model invented.
- No evidence retrieved -> the system returns one exact, fixed abstention message without ever
  calling the LLM provider. This is enforced in code (`generation/synthesizer.py`), not left to a
  prompt.
- Every source format is normalized to a **universal location** (`page:7`, `table:2/row:4`,
  `row:15`, `$[4]`, `lines:20-35`, `line:12`) so a citation always means something concrete
  regardless of source format.

`HANDOFF.md` has the project's origin story and cleanup history if you need it; it isn't required
reading for day-to-day work in the codebase.

## Stack

Pinned in `requirements.txt`: `llama-index-core==0.14.23` plus the Ollama/Chroma/BM25/file-reader
LlamaIndex integration packages, `chromadb`, `streamlit`, `pypdf`, `python-docx`, and
`pytesseract`+`pdf2image` for OCR. Local models via Ollama: `llama3.2:3b` (generation) +
`nomic-embed-text:latest` (embedding) — both required even when `LLM_PROVIDER=anthropic`, since
embedding always runs locally. Anthropic (`llama-index-llms-anthropic`) is wired in as an explicit,
opt-in comparison provider (`LLM_PROVIDER=anthropic` + `ANTHROPIC_API_KEY` in `.env`).

## Commands

```bash
# Setup
cp .env.example .env
pip install -r requirements.txt

# Pipeline, in order (each stage is a separate command deliberately: "ingest once, query many times")
python prepare_corpus.py --check-dependencies   # check optional OCR deps
python prepare_corpus.py                        # extract data/raw/* -> data/processed/document_records.jsonl
python scripts/ingest_documents.py               # rebuild Chroma + BM25 (needs Ollama running); --force skips the overwrite confirmation
python scripts/inspect_index.py --mode semantic|bm25|hybrid   # inspect ranking without generation
python scripts/run_evaluation.py retrieval --mode hybrid --top-k 4 --run-name <name>
python scripts/run_evaluation.py answers   --mode hybrid --top-k 4 --run-name <name>
streamlit run app.py

# Tests — fully offline, no network/Ollama/Tesseract required
python -m pytest -q
python -m pytest tests/test_retrieval.py::test_rrf_rewards_agreement_by_rank   # single test
```

## Architecture

Everything is wired through `src/support_copilot/config.py`'s `AppConfig` (loaded once via
`AppConfig.from_env()`), which every script, service, and test fixture takes as an explicit
parameter — there is no global state. `configure_llamaindex(config)` registers the LLM/embedding
model on LlamaIndex's global `Settings` as a side effect and must be called once per process before
any retrieval/generation call.

Pipeline stages and where to trace each one (from `README.md`, kept accurate — update this table if
the package layout changes):

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

Key mechanics worth knowing before touching retrieval or generation code:

- **Two-stage preparation/indexing split, not incremental.** `prepare_corpus.py` extracts raw files
  into `data/processed/document_records.jsonl` (source-of-truth normalized records, raw files never
  touched). `scripts/ingest_documents.py` does a **full rebuild** of both Chroma and BM25 from those
  records every time — this is deliberate, not a missing optimization (see
  `docs/THE_JOURNEY_TO_PRODUCTION.md`). Repeating it produces the same collection size and never
  duplicates nodes.
- **Index manifest is a compatibility gate.** Every successful `ingest()` writes
  `storage/index_manifest.json` (node count, corpus SHA-256, chunk policy, embedding model,
  `BM25_POLICY_VERSION`). `load_index`/`load_bm25` in `indexing/index_manager.py` refuse to run
  against a missing or version-mismatched manifest rather than silently searching stale term
  statistics — don't bypass this check.
- **Retrieval candidate pool vs. final top-k are different numbers, and both get clamped to corpus
  size.** `retrieve()` in `retrieval/retriever.py` always retrieves `resolve_candidate_k(top_k)` =
  `max(50, top_k * 5)` candidates per branch before fusing. That constant is a real diagnosed bug,
  not an arbitrary choice (F1 in `docs/known-findings.md`) — don't shrink it to "optimize" without
  reading that finding first. Separately, `load_bm25()` in `indexing/index_manager.py` clamps
  `similarity_top_k` to the manifest's `node_count`, because the `bm25s` backend hard-errors when
  asked for more results than the corpus contains — this matters on the shipped ~24-node sample
  corpus (smaller than the 50-candidate floor) and becomes a no-op once a corpus exceeds that floor.
- **Hybrid = semantic (Chroma) + BM25, fused with hand-rolled RRF, then reranked.**
  `reciprocal_rank_fusion` in `retriever.py` preserves each branch's original rank in node metadata
  (`semantic_rank`, `bm25_rank`) for UI/debugging — it clones nodes before mutating metadata so
  persisted source metadata is never touched.
- **Grounding contract is enforced in code, not just prompted.** `synthesizer.py`: no evidence ->
  return `ABSTENTION_MESSAGE` without calling the provider; `normalize_answer()` collapses any
  answer that *starts* with the abstention message back to the exact message, even if the model
  appended more text after it. Citations (`generation/citations.py`) are built entirely from
  retrieved Node metadata — never parsed out of model output. Read
  `docs/grounding-abstention-citations.md` before changing prompts or the synthesizer.
- **The contract stops fabrication; it doesn't guarantee the model uses good evidence.** With
  `llama3.2:3b` and several low-relevance evidence blocks mixed in (normal at `top_k=4` on the small
  sample corpus), the model has been observed to abstain even when the correct answer was retrieved
  and correctly citeable. That's a model-quality finding to measure with
  `scripts/run_evaluation.py answers` (its `false_abstentions` metric exists for exactly this), not
  something to silently fix by rewording the prompt.
- **Evaluation is pair-aware.** Ground truth is `(source, location)` pairs; a correct source at the
  wrong location does not satisfy an expectation (`manual.pdf` at `page:2` != `page:7`). Retrieval
  reports compute Hit@K, Recall@K, context precision, MRR; answer reports separately track citation
  marker validity, false answers, and false abstentions. The deterministic checks are a smoke
  signal, not a substitute for the human 0/1/2 rubric.
- **Services are the shared boundary.** `services/ingestion_service.py` and
  `services/query_service.py` are the only entry points `app.py` (Streamlit), `scripts/*.py`, and
  the evaluation runner should all go through — avoid calling `retrieval`/`generation`/`indexing`
  internals directly from a new interface.

## The sample corpus

`data/raw/` ships 3 small, support-flavored files for a fictional product ("Lumen Sync"):
`lumen_user_guide.md`, `support_faq.md`, `support_policy.txt` — sized to exercise the pipeline
(multi-file discovery, line-addressable citations across distinct sources), not to demonstrate
scale. `evaluation/datasets/evaluation_questions.json` has `(source, location)` ground truth
verified against these exact files' actual block boundaries (`ingestion/loaders/text_loader.py`
splits text on blank lines) — if you edit or replace any of them, re-verify the matching evaluation
cases rather than guessing new line ranges. These 3 files are explicitly un-ignored in `.gitignore`
so they ship with the repo; anything else dropped into `data/raw/` stays gitignored by default.

## When debugging, follow the failure boundaries in pipeline order

1. Preparation: unsupported/empty/invalid/duplicate file, OCR trigger, parser warning.
2. Node creation: missing metadata, unstable IDs, bad chunk boundary.
3. Index lifecycle: stale/missing manifest, empty Chroma, missing BM25.
4. Retrieval: expected pair absent or ranked too low in semantic/BM25/hybrid.
5. Grounding: evidence present but answer adds unsupported content, fails to cite it, or falsely abstains.
6. Evaluation: bad ground truth, misleading average, incomplete human rubric.

Only change prompts after confirming the necessary evidence was actually retrieved — most
"generation bugs" in this codebase turn out to be retrieval, corpus-size, or evaluation-data bugs
(see `docs/known-findings.md` for four real historical examples: RRF candidate-pool sizing, models
continuing to answer after saying "I don't know," wrong ground truth scoring correct retrieval as a
failure, PDF text-layer character artifacts — diagnosed against this project's original, larger
sample corpus, and still valid lessons even though that corpus was later swapped out).

## Honest current state

`docs/ARCHITECTURE_ASSESSMENT.md` scores this project 8/10 architecture, 8/10 code-level design,
4/10 as a production system — the low production score is intentional, not a defect: this is a
working prototype demonstrating grounded retrieval, and `docs/THE_JOURNEY_TO_PRODUCTION.md` is the
explicit, stage-by-stage plan for closing that gap (incremental ingestion, horizontal-scale vector
and keyword stores, orchestration, observability) without changing the retrieval/grounding logic
itself.

## Design docs (read before non-trivial changes in these areas)

- `docs/grounding-abstention-citations.md` — required reading before touching prompts or the synthesizer.
- `docs/known-findings.md` — real diagnosed bugs with symptom/diagnosis/fix/lesson; check here first when something looks like a new bug.
- `docs/DESIGN_QUESTIONS.md` — open architecture questions worth reasoning through before extending this project.
- `docs/ARCHITECTURE_ASSESSMENT.md` — scored self-assessment (architecture, code design, production-readiness).
- `docs/THE_JOURNEY_TO_PRODUCTION.md` — what's deliberately simplified at this scale vs. what changes at real scale (e.g. full-rebuild indexing).
