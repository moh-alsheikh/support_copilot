# Bi-Encoders and Cross-Encoders: Retrieve Wide, Rerank Narrow

How this project finds evidence today (a bi-encoder), what a cross-encoder reranker would add, and
why it targets the exact retrieval failures measured on the Raspberry Pi knowledge base. Written
2026-10-05, updated 2026-11-10: the cross-encoder is **implemented** — see *What was built* below.

## Two ways to score "is this passage relevant to this question?"

| | Bi-encoder | Cross-encoder |
|---|---|---|
| Input | Question and passage encoded **separately**, each into a vector | Question and passage fed in **together** as one input |
| Output | Two vectors, compared with cosine distance | One relevance score |
| What the model can see | Never sees both texts at once; works from compressed summaries | Every word of the question attends to every word of the passage |
| Precomputable? | Yes: passage vectors are computed once, at index time | No: must run once per (question, passage) pair, at question time |
| Speed | Searches millions of chunks in milliseconds | ~50 pairs per question is a practical budget on a CPU |
| Weakness | Blurs fine details: digits, negation, *which* product | Too slow to search a whole corpus |
| Role | First stage: find plausible candidates | Second stage: put the best candidates first |

The two are complementary, not alternatives. The standard pattern, and the one this project is
already structured for, is **retrieve wide, rerank narrow**: a fast first stage pulls a generous
candidate pool, and a precise second stage reorders it before anything reaches the LLM.

## The bi-encoder in this project today

There is no class named "bi-encoder". It is the embedding model plus how three components use it:

| Part | Who does it | Where |
|---|---|---|
| Runs the model (text → 768-number vector) | **Ollama**, serving `nomic-embed-text` on `localhost:11434` | Outside Python |
| Decides what to embed and when | **LlamaIndex** | Installed `llama-index-core` / `llama-index-embeddings-ollama` |
| Stores vectors, finds nearest neighbors | **Chroma** (HNSW, cosine) | `storage/chroma_db/` |
| Configuration only | **This project** | See below |

The configuration points:

- **Model:** `Settings.embed_model = OllamaEmbedding(model_name=config.embedding_model)` in
  `src/support_copilot/config.py` (`configure_llamaindex()`).
- **Passages, at index time:** `VectorStoreIndex(nodes, storage_context=...)` in `ingest()`,
  `src/support_copilot/indexing/index_manager.py`. This is most of the ~4 minute rebuild.
- **Distance:** `DISTANCE_SPACE = "cosine"` in `src/support_copilot/indexing/vector_store.py`.
- **Questions, at query time:** `load_index(config).as_retriever(similarity_top_k=candidate_k).retrieve(question)`
  in `retrieve()`, `src/support_copilot/retrieval/retriever.py`.

Passages are encoded once at indexing and the question separately at query time; they meet only as
a cosine comparison between finished vectors. How that distance becomes the displayed score is
covered in [`vector-similarity-scoring.md`](vector-similarity-scoring.md).

### Measured limits of the bi-encoder here

- **Digits blur.** For Q001 ("0 long flashes and 7 short flashes"), semantic search ranks the
  correct LED row (`Long flashes: 0 | Short flashes: 7 | Status: Kernel image not found`)
  **43rd**. Its top results are the code-table introduction and rows from a *different* LED table
  (the per-model power/activity summary): everything about "LED" scores alike, and the digits that
  identify the one right row barely register.
- **Two configuration gaps, separate from the reranker.** `nomic-embed-text` expects
  `search_query:` / `search_document:` prefixes; `OllamaEmbedding` supports them
  (`query_instruction`, `text_instruction`), but the project doesn't pass them. And each chunk's
  metadata (`content_hash`, `encoding`, `doc_id`, …) is embedded and BM25-indexed alongside its
  text.

Swapping the embedding model would not fix the digit problem: any bi-encoder compresses question
and passage separately.

## Where reranking sits in this pipeline

```
question
  → semantic search (top 50)  ┐
  → BM25 search (top 50)      ┴→ RRF fusion → [ RERANK ] → keep top 4 → [Source N] evidence → LLM answer
```

- The candidate pool already exists: `resolve_candidate_k()` in `retrieval/retriever.py` returns
  `max(50, top_k * 5)`, and fusion keeps the whole pool, so `rerank()` sees all ~50 candidates.
- The rerank step also exists: `rerank()` in `retrieval/reranker.py`. Today it adds **+0.001 per
  question word found in the chunk**, counting only words longer than 2 characters, so it ignores
  `0` and `7` exactly like the old BM25 tokenizer did.
- It runs only in hybrid mode, the default for the app, `ask.py`, and evaluation.

A cross-encoder would **replace the body of `rerank()`**. Nothing upstream (indexing, embeddings,
BM25, fusion) or downstream (generation, citations, the abstention contract) changes.

## Worked example: Q001

Ranks are real (hybrid mode, BM25 policy 2). **Cross-encoder scores are illustrative**; no
cross-encoder has been run on this project.

```
QUESTION: "What does it mean when my Raspberry Pi LED shows 0 long flashes and 7 short flashes?"

STEP 1 · Retrieve wide (unchanged)
  semantic top 50 ─┐
  BM25 top 50     ─┴─► RRF fusion ─► 50 candidates
     rank  1  lines:147-149  "Check LED warning codes ... pattern repeats after a two-second pause"
     rank  2  lines:200-202  "Long flashes: 3 | Short flashes: 1 | Status: SPI EEPROM error"
     rank  3  lines:208-210  "Long flashes: 3 | Short flashes: 3 | Status: I2C error"
     rank  4  lines:196-198  "Long flashes: 2 | Short flashes: 4 | Status: File signature/hash mismatch"
     ...
     rank 12  lines:164-166  "Long flashes: 0 | Short flashes: 7 | Status: Kernel image not found"  ◄ answer
     (today the top-4 cut happens here, so the answer never reaches the LLM)

STEP 2 · Rerank narrow (new)
  The cross-encoder scores each (question, candidate) pair:
     [question] + "Long flashes: 0 | Short flashes: 7 | ... Kernel image not found"  →  0.97
     [question] + "Check LED warning codes ... two-second pause"                      →  0.55
     [question] + "Long flashes: 3 | Short flashes: 3 | ... I2C error"                →  0.12
     ... 47 more pairs ...

  New order:  1 lines:164-166 (0.97) · 2 lines:147-149 (0.55) · 3 lines:208-210 (0.12) · ...

STEP 3 · Keep top 4 → [Source 1..4] → LLM (unchanged)
  "0 long and 7 short flashes means the kernel image was not found [Source 1]."
  [Source 1] = led-warning-flash-codes.md | lines:164-166   (from metadata, as today)
```

The bi-encoder compared two separately computed summaries in which "0 … 7" and "3 … 3" look
alike. The cross-encoder reads `0 long … 7 short` in the question and `0 | 7` in the row at the
same time, so the mismatch on every other row is visible.

## How it would help this project

Measured problems a cross-encoder targets, from the `bm25-token-fix` retrieval run (hybrid, top 4,
23 questions):

| Problem | Today | Why a reranker should help |
|---|---|---|
| **Q001** (LED code 0/7) | Correct row at rank 12: a miss | It's inside the 50-candidate pool; precise scoring can lift it into the top 4 |
| **Q007** (operating temperature) | Correct page at rank 6: a near miss | Same: in the pool, just below the cut |
| **Context precision 0.382** | Most of the 4 evidence blocks are irrelevant | Better ordering means fewer irrelevant blocks reach the LLM |
| **`llama3.2:3b` false refusals** | 4 refusals with the right evidence retrieved (Q003, Q011, Q014, Q015) | Hypothesis only: noisy evidence appeared to trigger refusals, so cleaner evidence may help. Needs the answer evaluation to confirm |

What it does **not** fix:

- **Near-miss questions** (45W supply, Case fan, Pi 500+ price). A reranker orders candidates by
  *relative* relevance, so it still puts the best available chunk (the 27W supply) first. Refusing
  stays the job of the abstention contract and the LLM. A minimum-score threshold could let weak
  matches be dropped before generation, but that's a separate design decision with its own risk of
  false refusals.
- **Evidence that was never retrieved.** It can only reorder the ~50 candidates it's given.
- **The automated answer checks' blind spot.** Citations attached to wrong claims still pass
  `answer_checks()`.

## What was built (2026-11-10)

`retrieval/reranker.py` scores every fused hybrid candidate (~50) with
`Xenova/ms-marco-MiniLM-L-6-v2` through **`fastembed`**, which runs on ONNX Runtime — already
installed for Chroma — so no PyTorch was needed. The model (80 MB, Apache-2.0) downloads once to
`storage/models/`; `RERANKER_MODEL` in `.env` selects it, and an empty value keeps fusion order.
Each candidate is scored on its embed view (title plus text, no hashes or IDs). The fusion score is
kept as `rrf_score` next to `semantic_rank` and `bm25_rank`. `rerank()` takes an injectable scorer
so tests run offline.

The `SentenceTransformerRerank` route below was not taken: the last PyTorch with an Intel-Mac
build (2.2.2) was built against NumPy 1.x, while this environment has NumPy 2.5; Ollama has no
rerank endpoint (404).

Measured (hybrid, top 4, 23 questions, together with title-only embedded metadata):

| | Hit@4 | Recall@4 | Precision | MRR |
|---|---|---|---|---|
| Before (word-bonus rerank, all metadata embedded) | 0.895 | 0.856 | 0.382 | 0.807 |
| After | **0.947** | **0.916** | **0.395** | **0.947** |

- **Q007** went from a miss to rank 1; Q015, Q019, Q022 moved to rank 1; no question got worse.
- **Q001 still misses:** the correct row rose from rank 12 to 7, but the MiniLM model doesn't
  reliably match "0 long and 7 short" to `Long flashes: 0 | Short flashes: 7` — small cross-encoders
  are weak at exact digit matching. A larger model (e.g. `bge-reranker-base`, 1 GB) is the next
  thing to try.
- **Latency:** the cross-encoder itself takes ~1.9 s for 50 candidates on this CPU (0.3 s for 20),
  plus a one-time model load. Small next to the ~50 s LLM answer.

## Implementation options considered

**First-choice route (not taken, see above): LlamaIndex's built-in reranker.** `SentenceTransformerRerank` ships in the
installed `llama-index-core` (`llama_index/core/postprocessor/sbert_rerank.py`). Three things to
know about it:

- **Its default model is the wrong kind.** The default, `cross-encoder/stsb-distilroberta-base`,
  is trained for sentence *similarity*, not search relevance. Pass a relevance-trained model
  explicitly, e.g. an MS MARCO cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2` class) or a
  `bge-reranker` model. Check each model card's license before choosing.
- **It defaults to `trust_remote_code=True`.** That lets a downloaded model run arbitrary code;
  set it to `False`.
- **New dependencies.** It needs `sentence-transformers` and PyTorch, which aren't installed today
  (hundreds of MB), plus a one-time model download. Whether Ollama can serve rerankers instead,
  avoiding PyTorch, is unverified and worth checking first.

**Expected cost on this laptop** (CPU-only i7-4770HQ, 16 GB), estimates to be measured:

| Model size | Added latency per question (50 pairs) | Note |
|---|---|---|
| MiniLM-class (~22M parameters) | ~1–2 s | Start here |
| `bge-reranker-base`-class (~280M) | several seconds | Only if the small model isn't precise enough |

Both are small next to the ~53 s it takes `llama3.2:3b` to write an answer.

**Design points for the change:**

- Replace the body of `rerank()` in `retrieval/reranker.py`; keep its signature, so `retrieve()`
  doesn't change.
- Keep `semantic_rank` and `bm25_rank` in node metadata (they're set before reranking), and add
  the rerank score, so the UI and reports can still explain an ordering.
- Hybrid scores will change scale again (from RRF sums to reranker scores). Update the hybrid row
  in [`vector-similarity-scoring.md`](vector-similarity-scoring.md).
- Keep tests offline: make the scorer injectable, so tests use a fake scorer and never download a
  model.

## How to decide whether it worked

1. Retrieval evaluation, all 23 questions, hybrid mode, before vs. after:
   `python scripts/run_evaluation.py retrieval --mode hybrid --top-k 4 --run-name <name>`.
   Success: Q001 and Q007 in the top 4, context precision up, no question's rank gets worse.
2. Measure the added latency per question.
3. Answer evaluation (`run_evaluation.py answers`) to test the false-refusal hypothesis. Score it
   with the human rubric, since the automated checks can't tell a wrong claim from a right one.

## Related

- [`vector-similarity-scoring.md`](vector-similarity-scoring.md) — how semantic, BM25, and hybrid
  scores are computed today.
- [`local-llm-model-selection.md`](local-llm-model-selection.md) — the generator-side comparison
  (`llama3.2:3b` vs. alternatives).
- [`../RAG_TECHNIQUE_COVERAGE.md`](../RAG_TECHNIQUE_COVERAGE.md) — scores the current reranker as
  partial (§3).
