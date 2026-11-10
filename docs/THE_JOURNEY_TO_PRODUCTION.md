# The Journey To Production

This project is intentionally built at demo scale: a full run holds every Document, every Node,
and every retrieved candidate in memory at once, and `prepare_corpus.py` re-extracts every file
on every invocation (indexing is incremental, but runs in one process against a local document
store and an in-memory BM25 index). That is the right choice when the
corpus is one manual plus a handful of sample files and the goal is to make every step
inspectable. It is the wrong choice for a real corpus measured in gigabytes or terabytes.

This document answers the natural next question a reviewer asks: *"What new operational problems
would Pinecone or pgvector introduce?"* — generalized to the whole pipeline, not only the vector
store. It is a reference for scaling this project up, not a spec for changes to the current code.
The retrieval logic here (RRF fusion, hybrid ranking, grounded synthesis, citation contracts)
does not change at scale. Only the plumbing around it does.

## Where This Project Hits A Wall

| Stage | This Project's Pattern | Breaks Down When |
|---|---|---|
| Raw storage | Local folder, `data/raw/`, scanned recursively | Corpus doesn't fit on one disk, or has many contributors |
| Change detection | Full re-extraction on every `prepare_corpus.py` run; file hashing only dedupes identical files, it doesn't skip unchanged ones across runs | Re-running preparation costs hours instead of seconds |
| Staging format | One `document_records.jsonl`, written and read as a single in-memory list (`processed_records.py`) | The file no longer fits in RAM to read or write |
| Extraction/OCR | Synchronous, one file at a time, local Tesseract | Thousands of scanned pages need extracting in parallel |
| Node/embedding build | `build_nodes()` holds every Document and every Node in memory before indexing | Corpus produces more nodes than RAM can hold at once |
| Vector index | `VectorStoreIndex(nodes, ...)` with `reset=True` — full collection replace every run | Index is too large to rebuild from scratch on every change |
| Keyword index | `BM25Retriever` persisted as one pickled file, loaded fully into RAM (`load_bm25`) | Term index no longer fits in memory |
| Orchestration | One synchronous script run by hand | Ingestion needs to run continuously, retry failures, and scale horizontally |
| Observability | One JSON report per run (`corpus_preparation.json`) | You need to know *why* document 40,000 of 2,000,000 failed, right now |

## The Production Pipeline

```text
Object storage (raw files)
  -> change detection (content-hash registry in a database)
  -> distributed extraction/OCR workers
  -> partitioned staging format (Parquet, not one JSONL)
  -> batched chunking + cached, batched embedding
  -> incremental upsert into a horizontally-scalable vector store
  -> incremental upsert into a horizontally-scalable keyword store
  -> hybrid retrieval + RRF + rerank (unchanged from this project's current pipeline)
  -> orchestrated as a retryable, observable pipeline, not a hand-run script
```

Everything below the "hybrid retrieval + RRF + rerank" line is exactly what this project already
implements. Everything above it is what changes.

## Stage By Stage

### 1. Raw Storage And Change Detection

Replace the local `data/raw/` folder and the JSON `document_register.py` file with object storage
plus a real database row per source file: `(source_path, content_hash, last_processed_at,
status)`. On each pipeline run, only files whose hash changed get reprocessed. This project already does
that at the indexing stage (a local `SimpleDocumentStore` of per-record hashes), but not yet at
extraction.

| Concern | Tool / Framework | Notes |
|---|---|---|
| Raw file storage | Amazon S3, Google Cloud Storage, Azure Blob Storage | Durable, versioned, cheap at scale; supports event notifications |
| Change-detection registry | PostgreSQL, SQLite (small scale), DynamoDB | One row per source file; hash + status + timestamps |
| Incremental upsert primitive | LlamaIndex `IngestionPipeline` + `docstore` (Redis/MongoDB/Postgres-backed) | Built-in doc-hash-based skip logic — already used by `index_manager.ingest()` with a local `SimpleDocumentStore`; production swaps in a shared, database-backed docstore |

### 2. Extraction And OCR

Don't shell out to a local OCR call per file synchronously, as `ingestion/ocr.py` does here for a
handful of files. Distribute extraction across workers, and use a managed or purpose-built
extraction service instead of raw Tesseract calls for anything beyond demo scale.

| Concern | Tool / Framework | Notes |
|---|---|---|
| Multi-format extraction | `unstructured.io` (open source) | Partitions PDFs/DOCX/images at scale, drop-in replacement for this project's `ingestion/loaders/` |
| Managed OCR/document AI | AWS Textract, Azure Document Intelligence, Google Document AI | Handles scanned pages, tables, forms without self-hosted OCR infrastructure |
| Parallel execution | Ray Data, Dask, Celery/RQ + a task queue | Each worker claims one file, extracts, writes its own output independently |

### 3. Normalized Staging Format

Replace the single `document_records.jsonl` — read and written as one in-memory list by
`processed_records.py` — with a partitioned, columnar format that can be queried and streamed
without loading the whole corpus into RAM.

| Concern | Tool / Framework | Notes |
|---|---|---|
| Staging format | Apache Parquet, partitioned by source or batch | Columnar, splittable, queryable directly without full load |
| Query engine over staged data | DuckDB, Apache Spark | Inspect/debug the corpus without a full ingestion run |
| If staying JSONL for simplicity | Shard into many files (one per N documents); stream-read with a generator | Keeps the format familiar but removes the single-file memory ceiling |

### 4. Chunking And Embedding

`build_nodes()` in this project materializes every Document and every Node in memory before
indexing starts. At scale, chunk and embed in streamed batches, and cache embeddings so an
unchanged chunk is never paid for twice.

| Concern | Tool / Framework | Notes |
|---|---|---|
| Batch/self-hosted embedding serving | Text Embeddings Inference (TEI), vLLM | GPU-batched embedding throughput, self-hosted |
| Managed batch embedding | OpenAI, Cohere, or Voyage batch endpoints | Cheaper and faster than per-request embedding calls |
| Embedding cache | LlamaIndex `IngestionCache` (Redis-backed) | Skips re-embedding unchanged chunks by content hash |

### 5. Vector Store

Chroma, as used in this project, is fine up to roughly low-millions of vectors on a single
machine, but it is not built for sharding or horizontal scale, and this project's `reset=True`
full-collection replace does not survive a corpus too large to rebuild from scratch.

| Concern | Tool / Framework | Notes |
|---|---|---|
| Self-hosted, sharded/replicated | Qdrant, Weaviate | Good filtering support, incremental upsert by ID |
| Purpose-built for very large vector counts | Milvus | Distributed by design |
| Already running Postgres | pgvector | One system of record instead of a separate vector service |
| Fully managed, no ops | Pinecone, Turbopuffer | Trade operational simplicity for a hosted dependency |

All four support incremental **upsert by ID** — the missing piece in this project's
`get_vector_store(config, reset=True)` pattern.

### 6. Keyword / BM25 Store

`BM25Retriever.from_persist_dir()` here loads one pickled index fully into memory. That does not
scale past what fits in RAM on one machine.

| Concern | Tool / Framework | Notes |
|---|---|---|
| Full-featured, sharded BM25 | OpenSearch, Elasticsearch | Native BM25, sharded, supports incremental indexing |
| Lighter self-hosted alternative | Meilisearch, Typesense | Simpler ops than an Elasticsearch cluster |
| Embeddable, on-disk, no server | Tantivy (Rust) | No separate service to run, index lives on disk instead of fully in RAM |

`QueryFusionRetriever` can wrap any of these the same way it wraps `BM25Retriever` today — the RRF fusion logic in
`retrieval/retriever.py` does not need to change, only the retriever it fuses.

### 7. Orchestration

Replace "run `prepare_corpus.py`, then run `ingest_documents.py`, by hand" with a scheduled or
event-driven pipeline that retries failures and scales workers independently per stage.

| Concern | Tool / Framework | Notes |
|---|---|---|
| Scheduled/DAG-based orchestration | Airflow, Prefect, Dagster | Each stage independently retryable and observable |
| Event-driven continuous ingestion | Object-storage upload event -> queue (SQS, Kafka) -> worker pool | For a corpus that grows continuously rather than in batch runs |

### 8. Observability

`corpus_preparation.json` here is the right idea — a structured report per run — but a one-shot
file does not scale to monitoring a pipeline that runs continuously across millions of documents.

| Concern | Tool / Framework | Notes |
|---|---|---|
| Structured, queryable run history | The same state database from Stage 1, plus structured logs | Query "why did document N fail" instead of grepping a report file |
| Metrics/dashboards | Grafana + Prometheus, or a hosted equivalent | Track throughput, failure rate, and retry counts over time |

## Consolidated Tool Reference

| Category | Current Choice (this project) | Production-Scale Options |
|---|---|---|
| Raw storage | Local folder (`data/raw/`) | S3 / GCS / Azure Blob |
| Change tracking | `document_register.py` JSON file | PostgreSQL / DynamoDB row per file |
| Extraction/OCR | `ingestion/loaders/`, local Tesseract | unstructured.io, AWS Textract, Azure/Google Document AI |
| Parallel processing | Single process | Ray Data, Dask, Celery/RQ |
| Staging format | One `document_records.jsonl` | Partitioned Parquet + DuckDB/Spark |
| Embedding | Direct calls during `ingest()` | TEI, vLLM, or provider batch APIs + `IngestionCache` |
| Vector store | Chroma | Qdrant, Weaviate, Milvus, pgvector, Pinecone |
| Keyword store | Persisted `BM25Retriever` pickle | OpenSearch, Elasticsearch, Meilisearch, Typesense, Tantivy |
| Orchestration | Hand-run scripts (`prepare_corpus.py`, `ingest_documents.py`) | Airflow, Prefect, Dagster, or event-driven queues |
| Observability | `corpus_preparation.json` | State DB + structured logs + Grafana/Prometheus |

## The Mental Model

Swap **"one process, everything in memory, full rebuild every time"** for **"many workers,
streamed batches, content-hash-driven incremental upserts, into horizontally-scalable stores."**
The chunking, hybrid retrieval, RRF fusion, and grounded-synthesis logic this project builds by
hand carries over unchanged — production scale is a plumbing problem, not a retrieval-quality
problem.

## Where This Connects

- [`ARCHITECTURE_ASSESSMENT.md`](ARCHITECTURE_ASSESSMENT.md) — scores this project's
  production-readiness at 4/10, by design; this document is the plan for closing that gap.
- [`known-findings.md`](known-findings.md) — F1 (candidate-pool sizing for RRF) and F3
  (pair-aware ground truth) still apply unchanged at any scale; they are retrieval-quality
  findings, not plumbing.
