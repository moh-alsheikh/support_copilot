# RAG Technique Coverage

How much of the broader advanced-RAG technique landscape this project actually implements, scored
item by item against the code rather than the docs.

- **Assessed:** 2026-09-22, against `master` at commit `8731610`.
- **Scoring:** each technique is one item. ✅ implemented = 1, 🟡 partial = 0.5, ❌ absent = 0.
  Section % = points ÷ items.
- **Evidence:** every ✅ and 🟡 names the file that implements it. Re-score this document when
  any of those files change.

## Summary

**Overall: 15.5 of 88 points — about 18%.** (Re-scored 2026-11-10 after incremental indexing.)

| # | Section | Items | Points | Coverage |
|---|---|---:|---:|---:|
| 1 | Document Ingestion & Chunking | 10 | 2.5 | 25% |
| 2 | Embedding & Representation | 9 | 2.5 | 28% |
| 3 | Indexing & Storage Architecture | 6 | 2.5 | 42% |
| 4 | Query Understanding & Transformation | 7 | 0 | 0% |
| 5 | Retrieval Strategies | 10 | 1.5 | 15% |
| 6 | Agentic RAG | 7 | 0 | 0% |
| 7 | Graph RAG & Structured RAG | 7 | 0 | 0% |
| 8 | Context Handling & Generation | 8 | 2.5 | 31% |
| 9 | Multimodal & Specialized RAG | 5 | 1 | 20% |
| 10 | Training & Fine-Tuning for RAG | 5 | 0 | 0% |
| 11 | Evaluation & Observability | 6 | 2.5 | 42% |
| 12 | Production & Enterprise | 8 | 0.5 | 6% |
| | **Total** | **88** | **15.5** | **18%** |

### How to read the 18%

The catalog is a menu, not a checklist. Many items are alternatives to each other: no system uses
every chunking strategy or every agent pattern. So 18% measures breadth, not quality.

What matters is where the coverage sits. The only sections above 30% are the ones this project
exists to demonstrate:

- **§3** hybrid retrieval with RRF
- **§8** citations and abstention
- **§11** retrieval evaluation

Four sections are at 0%: query transformation, agentic, graph, and training. Production is at 6%. That
matches the scope described in [`ARCHITECTURE_ASSESSMENT.md`](ARCHITECTURE_ASSESSMENT.md): the
project is a verifiable-grounding prototype, not a survey of techniques.

## Detail by section

### 1. Document Ingestion & Chunking — 25%

| Technique | Status | Evidence |
|---|:---:|---|
| Fixed-size with overlap | ✅ | `ingestion/parsers.py`: `SentenceSplitter`, 512-token chunks, 64-token overlap (12.5%) |
| Recursive / structure-aware chunking | 🟡 | Loaders pre-split by each format's structure: PDF by page, text by blank-line block, DOCX by paragraph or table row, CSV by row, JSON by path. No splitting on Markdown headers, HTML, or code ASTs |
| Semantic chunking | ❌ | — |
| Propositional chunking | ❌ | — |
| Late chunking | ❌ | — |
| Contextual retrieval | ❌ | — |
| Hierarchical / parent-child chunking | ❌ | — |
| Small-to-big | ❌ | — |
| Metadata enrichment | 🟡 | `ingestion/document_register.py`: title derived from the filename, `doc_id`, and manually entered topic/year/source fields. Nothing is LLM-extracted (no summaries, entities, or hypothetical questions) |
| Table / layout-aware parsing | 🟡 | DOCX tables become `table:T/row:R` records; CSV rows are labelled with their headers. PDFs use the `pypdf` text layer only, with no layout or table recovery |

### 2. Embedding & Representation — 28%

| Technique | Status | Evidence |
|---|:---:|---|
| Bi-encoder vs cross-encoder | ✅ | Bi-encoder retrieval with `nomic-embed-text` via Ollama. There is no cross-encoder anywhere (see §3, re-ranking) |
| Asymmetric embeddings | ❌ | Nomic expects `search_query:` / `search_document:` prefixes, but `config.py` builds `OllamaEmbedding(model_name=...)` without `query_instruction` / `text_instruction`, so queries and documents are embedded identically |
| Instruction-tuned embedders | 🟡 | Nomic is instruction-tuned but is used without its task prefixes (row above) |
| Matryoshka (MRL) | ❌ | — |
| Late interaction (ColBERT, ColPali) | ❌ | — |
| Sparse-dense hybrid | ✅ | BM25 (`bm25s`) plus dense vectors (Chroma), fused in `retrieval/retriever.py` |
| Multi-modal embeddings | ❌ | — |
| Code & domain-specific embedders | ❌ | — |
| Quantization | ❌ | — |

### 3. Indexing & Storage Architecture — 42%

| Technique | Status | Evidence |
|---|:---:|---|
| Vector index types | ✅ | HNSW via Chroma, with `hnsw:space="cosine"` set explicitly in `indexing/vector_store.py` |
| Hybrid search (BM25 + vector, RRF) | ✅ | Hand-rolled RRF (`rank_constant=60`) in `retrieval/retriever.py`. Candidate pool of `max(50, top_k * 5)` per branch; each branch's rank is kept in node metadata for debugging |
| Re-ranking pipeline | 🟡 | 50+ candidates are fused, but `retrieval/reranker.py` only adds +0.001 per query term (longer than 2 characters) found in the chunk. It's a lexical tiebreaker, not a cross-encoder |
| Hierarchical index (RAPTOR) | ❌ | — |
| Graph index | ❌ | — |
| Multi-index routing | ❌ | — |

### 4. Query Understanding & Transformation — 0%

None of the 7 items: rewriting, multi-query, HyDE, step-back prompting, decomposition, expansion,
or intent routing. The user's question is embedded and searched exactly as typed.

### 5. Retrieval Strategies — 15%

| Technique | Status | Evidence |
|---|:---:|---|
| Naive RAG | ✅ | `--mode semantic`: single-shot vector search |
| Advanced RAG | 🟡 | `--mode hybrid`: hybrid search plus re-ranking, but with the lexical reranker from §3 |
| Modular RAG | ❌ | Retrieval mode is a string branch inside `retrieve()`, not composable components (also noted in `ARCHITECTURE_ASSESSMENT.md`) |
| Adaptive RAG | ❌ | Retrieval always runs |
| Self-RAG | ❌ | — |
| Corrective RAG (CRAG) | ❌ | — |
| Speculative RAG | ❌ | — |
| RAG-Fusion | ❌ | RRF here fuses two retrievers over one query, not multiple query variants |
| LongRAG / long-context | ❌ | — |
| MemoRAG | ❌ | — |

### 6. Agentic RAG — 0%

None of the 7 items. Retrieval is a single fixed pass with no loop, tool choice, planner, or memory.

### 7. Graph RAG & Structured RAG — 0%

None of the 7 items. There is no knowledge graph, summary tree, entity linking, or SQL path.

### 8. Context Handling & Generation — 31%

| Technique | Status | Evidence |
|---|:---:|---|
| Context compression | ❌ | — |
| Contextual compression retrieval | ❌ | — |
| Reordering (lost-in-the-middle) | ❌ | Evidence is passed to the model in rank order |
| Citation & attribution | ✅ | Numbered `[Source N]` evidence blocks. Citation rows are built from retrieved node metadata, never parsed from model output. Numbering stays stable when two chunks share a location (`generation/synthesizer.py`, `generation/citations.py`) |
| Answerability check | ✅ | Empty retrieval returns the fixed `ABSTENTION_MESSAGE` without calling the LLM. Both the QA and refine prompts require exact abstention. `normalize_answer()` collapses answers that start with the abstention message and then answer anyway |
| Faithfulness / groundedness | 🟡 | Evidence-only prompt rules, abstention normalization, and citation-marker validity checks. No per-claim verification |
| Summarization-augmented generation | ❌ | — |
| Chain-of-Verification | ❌ | — |

### 9. Multimodal & Specialized RAG — 20%

| Technique | Status | Evidence |
|---|:---:|---|
| Multimodal RAG | 🟡 | Images and scanned PDF pages are OCR'd to text at ingestion (`ingestion/ocr.py`). Retrieval and generation are text-only |
| Visual document retrieval | ❌ | — |
| Table RAG | 🟡 | DOCX and CSV tables are retrievable row by row, flattened to text. No table-QA model |
| Code RAG | ❌ | — |
| Voice / audio RAG | ❌ | — |

### 10. Training & Fine-Tuning for RAG — 0%

None of the 5 items. All models are used off the shelf.

### 11. Evaluation & Observability — 42%

| Technique | Status | Evidence |
|---|:---:|---|
| Retrieval metrics | ✅ | Hit@K, Recall@K, context precision, and MRR over pair-aware `(source, location)` ground truth (`evaluation/evaluators.py`). nDCG is missing |
| Generation metrics | 🟡 | Deterministic `answer_checks()`: expected answer contained, abstention correctness, false answers, false abstentions, invalid citation markers, citations not in retrieved evidence, expected evidence cited. No semantic faithfulness or relevancy scoring |
| RAG Triad | ❌ | — |
| Frameworks (RAGAS, ARES, TruLens, DeepEval) | ❌ | — |
| Failure taxonomy | 🟡 | Six-stage failure boundaries (README), real diagnosed cases in `known-findings.md`, and a false-abstention (wrongful refusal) metric |
| Human vs LLM-as-judge | 🟡 | `manual_rubric()` exports human 0–2 scores for correctness, faithfulness, completeness, and citation accuracy. No LLM judge |

### 12. Production & Enterprise — 6%

| Technique | Status | Evidence |
|---|:---:|---|
| Freshness & real-time indexing | 🟡 | Incremental indexing: `index_manager.ingest()` runs LlamaIndex's `IngestionPipeline` with a local `SimpleDocumentStore` and `UPSERTS_AND_DELETE`, embedding only new or changed records and deleting removed ones. No change-data capture or real-time trigger; runs are manual, and extraction still reprocesses every file |

The other 7 items (caching, invalidation, access control, scaling, cost tiering, personalization,
privacy-preserving search) are covered only as a plan in
[`THE_JOURNEY_TO_PRODUCTION.md`](THE_JOURNEY_TO_PRODUCTION.md).

## Judgment calls

Four scores could reasonably go either way. Each is stated here so the total can be audited:

- **Bi-encoder vs cross-encoder = 1.** The bi-encoder half exists; the cross-encoder half is missing
  and is counted against §3 re-ranking instead.
- **Modular RAG = 0.** Retrieval modes are selectable, but only through a string branch, not
  swappable components.
- **Table RAG = 0.5.** Tables are retrievable at row level, but there's no table-aware reasoning.
- **Retrieval metrics = 1.** Four of the five named metrics exist; only nDCG is missing.

## Highest-leverage next steps

These strengthen what the project already claims to do, rather than adding breadth:

1. **Use Nomic's task prefixes.** Pass `query_instruction="search_query:"` and
   `text_instruction="search_document:"` to `OllamaEmbedding` in `configure_llamaindex()`. It's a
   small code change, but existing vectors were embedded without the prefix, so it needs an
   `ingest_documents.py --force` rebuild. Fixes the §2 asymmetric-embedding gap.
2. **Replace the lexical reranker with a cross-encoder.** The 50-candidate pool is already there to
   rerank. This turns §3 re-ranking and §5 Advanced RAG into full implementations.
3. **Add nDCG to `evaluation/evaluators.py`.** It completes the §11 retrieval metrics set.

## Related

- [`ARCHITECTURE_ASSESSMENT.md`](ARCHITECTURE_ASSESSMENT.md) — scored quality assessment
  (architecture, design, production readiness). This document scores breadth instead.
- [`THE_JOURNEY_TO_PRODUCTION.md`](THE_JOURNEY_TO_PRODUCTION.md) — plan for the §12 gaps.
- [`BACKLOG.md`](BACKLOG.md) — features queued for implementation.
- [`learning/vector-similarity-scoring.md`](learning/vector-similarity-scoring.md) — how the §3
  retrieval scores are actually computed.
