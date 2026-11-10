# Retrieval Scoring: What Semantic, BM25, and Hybrid Scores Actually Mean

`python scripts/inspect_index.py --mode semantic|bm25|hybrid` prints a `score` per result. It is
easy to mis-read that number as a percent match, or to assume it means the same thing across
modes. It doesn't. This note explains what each one actually is, verified against this project's
own running index and the installed library source rather than assumed from general knowledge of
Chroma or LlamaIndex.

## Two independent config layers

Two separate systems are involved, and neither is aware of the other's configuration:

1. **Chroma decides what "distance" *is*** — controlled by `hnsw:space`, set as collection metadata
   at creation time (`chromadb/api/configuration.py`), and immutable afterward: changing it means
   recreating the collection, not updating existing state.
2. **LlamaIndex decides how to turn that distance into a "score"** — in
   `llama_index/vector_stores/chroma/base.py`'s `_query()`, unconditionally:

   ```python
   similarity_score = math.exp(-distance)
   ```

   There is no branch on the collection's configured space. LlamaIndex applies the exact same
   formula whether Chroma handed back squared-L2 distance, cosine distance, or raw inner product —
   it just trusts "smaller distance = more similar" and exponentiates the sign away.

Until this was investigated, this project's `indexing/vector_store.py`'s `get_collection()` never
set `hnsw:space`, so Chroma silently fell back to its default. See **Resolution** below for what
changed.

## Chroma's three distance metrics

Formulas below are Chroma's own reference implementation
(`chromadb/utils/distance_functions.py`), which its docstring states matches the hnswlib spec:

| `hnsw:space` | formula | range | used by this project? |
|---|---|---|---|
| `"l2"` (default) | `‖x − y‖²` (squared Euclidean) | `[0, ∞)` | no longer — was the silent default before this fix |
| `"cosine"` | `1 − cos_sim(x, y)` | `[0, 2]` | **yes, explicitly, as of this fix** |
| `"ip"` | `1 − dot(x, y)` (raw, **not** normalized) | unbounded, either sign | not used |

## General rule of thumb: do you need to set this?

Yes, in general — set `hnsw:space="cosine"` explicitly rather than leaving Chroma on its `"l2"`
default, for basically any text/RAG use case:

- Most embedding models (OpenAI, Cohere, `nomic-embed-text`, sentence-transformers/BGE/E5) return
  normalized vectors, so `l2` happens to rank identically to `cosine` when that holds — proven
  below for this project's model. But that's a silent assumption about *your specific model*, not
  something Chroma enforces or checks, and making it explicit costs nothing: same rankings when the
  assumption holds, correct rankings even if it stops holding later (e.g. you swap embedding
  providers).
- Reach for `"ip"` (raw inner product) instead only for a model specifically trained for
  dot-product retrieval — some MIPS-style dual-encoder or recommendation setups where embedding
  magnitude is intentionally meaningful (e.g. popularity-weighted). Uncommon for off-the-shelf text
  embedding models.
- Leaving it at `"l2"` mainly makes sense outside text search entirely — domains like image/feature
  vector similarity where raw Euclidean geometry is the actual meaningful distance.
- The practical habit: default to `cosine` and set it explicitly at collection-creation time. Don't
  leave it unset and inherit whatever Chroma's default happens to be.

## What this means in this project (verified, not theoretical)

Pulled a real query/document embedding pair from the live `nomic-embed-text` model and checked
their norms directly:

```
norm(query embedding) = 1.0000002470495573
norm(doc embedding)   = 0.9999997309507457
```

`nomic-embed-text` returns **unit-normalized vectors**. That matters because of an exact identity
that only holds for unit vectors:

```
‖x − y‖² = 2 · (1 − cos_sim(x, y))    i.e.    l2_squared = 2 × cosine_distance
```

Checked the ratio on real data from this project's index:

```
l2_squared = 0.8606982855963348
cosine_distance (1 - cos_sim) = 0.43034915226561377
ratio = 1.999999956001092   # exactly 2, within float precision
```

And since `dot(x, y) = cos_sim(x, y)` for unit vectors too, `ip` distance collapses to the same
quantity as `cosine` distance here as well.

**Conclusion: on this embedding model, all three `hnsw:space` settings produce identical
rankings.** They're just different monotonic rescalings of the same underlying cosine similarity.
Switching `hnsw:space` to `"cosine"` would change the *absolute* score numbers `inspect_index.py`
prints (to a more familiar, tightly bounded range) — it would not reorder a single result.

### The actual `inspect_index.py --mode semantic` scores, before and after

For the query "How do I reset my password?" against this project's sample knowledge base — same 4
results, same ranking, on the `"l2"` default (before this fix) vs explicit `"cosine"` (after):

| result | raw L2 distance | score under `"l2"` (before) | score under `"cosine"` (after) |
|---|---|---|---|
| password reset (correct match) | 0.647005 | 0.523612 | 0.723610 |
| cancel subscription | 1.007584 | 0.365100 | 0.604235 |
| change billing cycle | 1.012898 | 0.363165 | 0.602632 |
| account deletion | 1.016178 | 0.361976 | 0.601644 |

Ranking is identical before and after (smallest distance → highest score, always — `exp` is
monotonic, and `l2_squared = 2·cosine_distance` for this model, as proven above). What changed is
only the absolute numbers: `"cosine"` compresses the achievable range to `exp(-2)..exp(0) ≈
0.135..1.0`, which is why the post-fix scores sit closer together in absolute terms but preserve
the same relative gaps and the same winner.

## Resolution

`src/support_copilot/indexing/vector_store.py` now sets `hnsw:space` explicitly via a
`DISTANCE_SPACE = "cosine"` constant, passed as `metadata={"hnsw:space": DISTANCE_SPACE}` to
`get_or_create_collection()`. This closes the incidental-correctness gap described above:
ranking no longer depends on an unstated assumption that the embedding model normalizes its
output — it depends on an explicit, documented choice instead.

Two things worth knowing about how this actually took effect, both confirmed against the live
index rather than assumed:

- `hnsw:space` is fixed at collection-creation time (`is_static=True` in Chroma's own
  configuration schema). Passing new `metadata` to `get_or_create_collection()` for a collection
  that **already exists** is silently accepted and silently ignored — no error, no change. Verified
  directly: calling it against this project's pre-fix collection printed `metadata: None`
  afterward, unchanged. The setting only takes effect on a fresh collection, which is exactly what
  `scripts/ingest_documents.py --force` produces (`index_manager.ingest()` always deletes then
  recreates — see `docs/THE_JOURNEY_TO_PRODUCTION.md` on why this project rebuilds fully rather
  than incrementally).
- After running `--force` here, `col.metadata` read back as `{'hnsw:space': 'cosine'}`, the scores
  above matched the "after" column exactly, and `scripts/run_evaluation.py retrieval --mode
  semantic` still scored Hit@K = Recall@K = MRR = 1.0 — confirming the ranking guarantee proven
  above held in practice, not just in the math.

**If you pull this change without rebuilding, nothing breaks** — `get_collection()` still returns
a working collection either way — but an existing local index keeps running on whatever space it
was originally built with until it's rebuilt. (Since incremental ingestion, `DISTANCE_SPACE` is part
of the vector policy recorded in the manifest, so changing it triggers a full rebuild
automatically on the next `ingest_documents.py` run.)

## BM25 scoring — a different mechanism entirely

Unlike semantic mode, BM25 has no distance-to-score transform layer. Confirmed directly from the
installed `llama-index-retrievers-bm25` source (`llama_index/retrievers/bm25/base.py:258`):

```python
nodes.append(NodeWithScore(node=node, score=float(score)))
```

`float(score)` is a type cast, not a transform. The `score` comes straight out of
`bm25s.BM25.retrieve()` (line 237) — a separate library from Chroma entirely — computing the
native Okapi BM25 formula (term frequency × inverse document frequency × length normalization)
directly as a "higher = more relevant" number. There is no Chroma involved in `--mode bm25` at
all; `bm25s` is a standalone term index, persisted separately under `storage/bm25/`.

| | semantic | bm25 |
|---|---|---|
| scoring library | Chroma (distance) + LlamaIndex (transform) | `bm25s` only |
| transform LlamaIndex applies | `exp(-distance)` | none — direct `float()` cast |
| this project's own code | 0 (pure pass-through) | 0 (pure pass-through) |
| score range | bounded `[0.135, 1.0]` (cosine, post-fix) | unbounded, corpus/term-dependent |
| meaning | monotonic w.r.t. cosine similarity | native BM25 relevance weight |

Both modes share one thing: this project's own code (`retrieve()`, `inspector.py`) never computes
or rescales either score — it's pure delegation and printing in both cases. But the number of
transform steps between "library output" and "what you see" differs (one for semantic, zero for
BM25), and the two scales were never comparable, even before the `hnsw:space` fix. Don't compare a
`bm25` score to a `semantic` score, or treat either as "percent relevance."

### A related finding: `BM25Retriever`'s own top-k guard doesn't cover this project's usage

`BM25Retriever.__init__` (`llama_index/retrievers/bm25/base.py:114-126`) already has a built-in
safety guard: if the constructor's `similarity_top_k` argument exceeds the indexed document count,
it logs a warning and clamps automatically, rather than letting `bm25s` raise. But that guard only
runs *inside* `__init__`, checked against the constructor argument.

This project's `indexing/index_manager.py::load_bm25()` doesn't go through that path — it calls
`BM25Retriever.from_persist_dir(path)` (construction with a different default `similarity_top_k`)
and then does:

```python
retriever.similarity_top_k = min(top_k, manifest["node_count"])
```

a **direct attribute reassignment after construction**, which bypasses `__init__`'s guard entirely
— reassigning an attribute doesn't re-run constructor logic. That's exactly why `bm25s` raised
`ValueError: k of 50 is larger than the number of available scores, which is 24` the first time
this project's small sample knowledge base was queried in hybrid/bm25 mode (candidate pool floor
of 50 vs. a 24-node corpus), and why the manual `min(top_k, manifest["node_count"])` clamp added to
`load_bm25()` was necessary rather than redundant with something the library already handled.

## Reading scores across retrieval modes

Summary, with detail sections above for semantic and bm25:

| mode | score is | range | own code computes it? |
|---|---|---|---|
| **semantic** | `exp(-distance)`, distance = Chroma cosine distance | `[0.135, 1.0]` | no — pass-through |
| **bm25** | raw `bm25s` Okapi BM25 weight | unbounded, corpus-dependent | no — pass-through |
| **hybrid** | Reciprocal Rank Fusion sum, `retrieval/retriever.py`'s `reciprocal_rank_fusion()`: `Σ 1 / (60 + rank)` per branch | typically ~0.01–0.03 | **yes** — this is the one score this project actually computes itself, from rank position, discarding both raw scores above |

None of the three scales are comparable to each other or to "percent relevance."
