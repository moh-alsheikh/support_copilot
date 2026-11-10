# Architecture Assessment

A short, scored rating of this project across three dimensions: architecture, code-level design,
and readiness as a startup/production project. Companion to `THE_JOURNEY_TO_PRODUCTION.md`, which
covers the production gap in depth — this document is the summary judgment; that one is the plan.

## Architecture: 8/10

The project is a layered pipeline — `ingestion/` → `indexing/` → `retrieval/` → `generation/` →
`evaluation/`, fronted by a `services/` façade shared by the CLI scripts and the Streamlit UI.
Each package only imports from the stage below it. Strengths: explicit DTOs (`models.py`) keep
the UI independent of LlamaIndex's internal types, `AppConfig` is passed as immutable data rather
than read from globals, and persisted state is guarded by a fingerprinted manifest
(`index_manager.py`) so a stale or incompatible index fails loudly instead of returning quietly
wrong rankings.

Points held back: `indexing/`, `retrieval/`, and `generation/` import LlamaIndex types directly
rather than behind an interface, so a framework swap would still ripple across multiple packages —
this is a layered/service-façade architecture, not a full hexagonal/ports-and-adapters design. Indexing
is incremental (LlamaIndex `IngestionPipeline` with a local document store), but BM25 is still
rebuilt in full each run and preparation still re-extracts every file.

## Design (Code-Level): 8/10

Strengths: a genuinely pure functional core — `reciprocal_rank_fusion()`, `rerank()`,
`citations_from_nodes()`, `normalize_answer()` — none of it touches I/O, so retrieval-quality
logic is unit-testable without mocking an LLM, embedding model, or vector store. Per-file failure
isolation (`LoadResult`, `FilePreparationStatus`) means one bad PDF doesn't cancel the whole
corpus-preparation run. Content-hash identity (`stable_record_id`, `corpus_fingerprint`) gives the
pipeline idempotency. Model output contracts (abstention normalization) are enforced in code, not
left to the prompt alone — directly implementing finding F2 in [`known-findings.md`](known-findings.md).

Points held back: retrieval-mode selection (`mode="semantic"/"bm25"/"hybrid"`) is a string branch
rather than a real `Strategy` object — a reasonable simplification at this scale, but the exact
seam where a polymorphic interface would go if strategies grew. A few stages
(`build_nodes()`, `load_records()`) hold the entire corpus in memory at once — acceptable for a
demo-scale corpus, not a pattern to copy at real scale.

## As A Startup/Production Project: 4/10

This is the expected score, not a flaw — the project is a working prototype, not a shipped
product. Indexing is incremental, but preparation still re-extracts every file and BM25 is rebuilt in
full on each run; no horizontal scale on Chroma or the pickled BM25 index, no
retry/orchestration layer, local-only Ollama and Chroma, single-process throughout.
`THE_JOURNEY_TO_PRODUCTION.md` exists precisely because closing this gap is real, deliberate
follow-on work — not an oversight in this codebase.

## Overall

For what it is built to be — a demonstration of a genuinely grounded, citation-verifiable RAG
pipeline built with LlamaIndex — **8/10**. The architecture and code-level design are solid
enough to build on directly; the production gap is real but intentional, and it's the explicit
subject of `THE_JOURNEY_TO_PRODUCTION.md` rather than something this document treats as a defect.
