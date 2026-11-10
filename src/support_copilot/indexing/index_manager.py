"""Build matching Chroma and BM25 indexes with a verifiable storage manifest.

Vector indexing is incremental: LlamaIndex's `IngestionPipeline` keeps a document store of
per-document hashes, so a run only splits and embeds new or changed documents and, with
`UPSERTS_AND_DELETE`, removes the vectors of documents that disappeared. That keeps the guarantee
the old full rebuild existed for (deleted records leave no stale vectors) without re-embedding an
unchanged corpus. BM25 has no incremental primitive and is cheap, so it is rebuilt from every node
currently in Chroma. A full rebuild still happens on request, when there is no document store to
diff against, or when the vector policy (chunking, embedding model, distance, chunk IDs) changed,
since those invalidate existing vectors without changing any document hash.

A manifest records the corpus fingerprint and indexing policy so query-time code can reject an
incompatible BM25 artifact instead of returning quietly misleading rankings.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.retrievers.bm25 import BM25Retriever

from support_copilot.indexing.vector_store import DISTANCE_SPACE, get_collection, get_vector_store
from support_copilot.ingestion.document_adapter import EMBEDDED_METADATA_KEYS, load_documents
from support_copilot.ingestion.parsers import build_parser

BM25_POLICY_VERSION = 2
# bm25s's default pattern (\b\w\w+\b) drops one-character tokens, so digits that distinguish
# otherwise-identical rows ("0 long, 7 short flashes", "Pi 5" vs "Pi 4") are never matched.
BM25_TOKEN_PATTERN = r"(?u)\b\w+\b"
MANIFEST_NAME = "index_manifest.json"
PIPELINE_DIR = "ingestion_pipeline"
DOCSTORE_NAME = "docstore.json"
# Bump when chunk IDs or how documents become nodes change in a way existing vectors can't absorb.
CHUNK_ID_SCHEME = 1


@dataclass
class IngestionReport:
    """What one ingestion run did, for the CLI and tests."""

    mode: str
    reason: str
    documents: int
    added: int
    changed: int
    removed: int
    unchanged: int
    nodes_embedded: int
    nodes_total: int


def vector_policy(config) -> dict:
    """Settings that, if changed, make every existing vector stale even when no document changed."""

    return {
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "embedding_model": config.embedding_model,
        "distance_space": DISTANCE_SPACE,
        "chunk_id_scheme": CHUNK_ID_SCHEME,
        "embedded_metadata_keys": list(EMBEDDED_METADATA_KEYS),
    }


def pipeline_dir(config) -> Path:
    """Return the directory holding the ingestion pipeline's document store."""

    return config.storage_dir / PIPELINE_DIR


def manifest_path(config) -> Path:
    """Return the generated index manifest path."""

    return config.storage_dir / MANIFEST_NAME


def corpus_fingerprint(nodes) -> str:
    """Hash stable node identities and text so you can identify the indexed corpus."""

    digest = hashlib.sha256()
    for node in sorted(nodes, key=lambda item: item.node_id):
        digest.update(node.node_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(node.get_content().encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def has_existing_indexes(config) -> bool:
    """Detect generated state that a full rebuild will replace."""

    bm25_path = config.storage_dir / "bm25"
    return (
        bm25_path.exists()
        or manifest_path(config).exists()
        or pipeline_dir(config).exists()
        or get_collection(config).count() > 0
    )


def write_manifest(config, nodes) -> Path:
    """Persist index policy and corpus identity beside generated storage."""

    config.storage_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "bm25_policy_version": BM25_POLICY_VERSION,
        "node_count": len(nodes),
        "corpus_sha256": corpus_fingerprint(nodes),
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "embedding_model": config.embedding_model,
        "vector_policy": vector_policy(config),
    }
    path = manifest_path(config)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_manifest(config) -> dict:
    """Load and validate the generated index compatibility contract."""

    path = manifest_path(config)
    if not path.exists():
        raise RuntimeError("The index manifest is missing. Run: python scripts/ingest_documents.py")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("bm25_policy_version") != BM25_POLICY_VERSION:
        raise RuntimeError("The BM25 policy changed. Rebuild indexes with ingest_documents.py.")
    return payload


def _full_rebuild_reason(config, docstore_path: Path) -> str | None:
    """Explain why existing vectors can't be updated incrementally, or None if they can."""

    if get_collection(config).count() == 0:
        return "no existing vector index"
    if not docstore_path.exists():
        return "no ingestion document store (index was built by full-rebuild code)"
    path = manifest_path(config)
    if not path.exists():
        return "index manifest is missing"
    recorded = json.loads(path.read_text(encoding="utf-8")).get("vector_policy")
    if recorded != vector_policy(config):
        return f"vector policy changed (was {recorded}, now {vector_policy(config)})"
    return None


def _rebuild_bm25(config, nodes) -> None:
    bm25_path = config.storage_dir / "bm25"
    if bm25_path.exists():
        shutil.rmtree(bm25_path)
    BM25Retriever.from_defaults(
        nodes=nodes, similarity_top_k=config.top_k, token_pattern=BM25_TOKEN_PATTERN
    ).persist(str(bm25_path))


def ingest(config, *, full_rebuild: bool = False) -> IngestionReport:
    """Update Chroma incrementally from normalized records, then rebuild BM25 and the manifest."""

    documents = load_documents(config)
    if not documents:
        raise ValueError("No prepared records were found. Inspect corpus preparation warnings.")

    docstore_path = pipeline_dir(config) / DOCSTORE_NAME
    reason = "requested" if full_rebuild else _full_rebuild_reason(config, docstore_path)
    full = reason is not None
    if full and pipeline_dir(config).exists():
        shutil.rmtree(pipeline_dir(config))
    docstore = SimpleDocumentStore() if full else SimpleDocumentStore.from_persist_path(str(docstore_path))

    current = {document.doc_id: document.hash for document in documents}
    previous = {doc_id: hash_ for hash_, doc_id in docstore.get_all_document_hashes().items()}
    added = sum(doc_id not in previous for doc_id in current)
    changed = sum(doc_id in previous and previous[doc_id] != hash_ for doc_id, hash_ in current.items())
    removed = sum(doc_id not in current for doc_id in previous)

    vector_store = get_vector_store(config, reset=full)
    pipeline = IngestionPipeline(
        transformations=[build_parser(config), Settings.embed_model],
        vector_store=vector_store,
        docstore=docstore,
        docstore_strategy=DocstoreStrategy.UPSERTS_AND_DELETE,
        disable_cache=True,
    )
    # store_doc_text must stay on: UPSERTS_AND_DELETE removes documents with docstore.delete_document(),
    # which raises for documents stored without text.
    embedded = pipeline.run(documents=documents)
    pipeline.persist(str(pipeline_dir(config)), docstore_name=DOCSTORE_NAME)

    nodes = sorted(vector_store.get_nodes(None), key=lambda node: node.node_id)
    if not nodes:
        raise ValueError("No searchable nodes were produced. Inspect corpus preparation warnings.")
    _rebuild_bm25(config, nodes)
    write_manifest(config, nodes)
    return IngestionReport(
        mode="full" if full else "incremental",
        reason=reason or "",
        documents=len(documents),
        added=added,
        changed=changed,
        removed=removed,
        unchanged=len(current) - added - changed,
        nodes_embedded=len(embedded),
        nodes_total=len(nodes),
    )


def load_index(config):
    """Reconnect to Chroma and fail clearly when indexing has not run."""

    load_manifest(config)
    if get_collection(config).count() == 0:
        raise RuntimeError("The vector index is empty. Run: python scripts/ingest_documents.py")
    return VectorStoreIndex.from_vector_store(get_vector_store(config))


def load_bm25(config, *, top_k: int):
    """Load the persisted term index only when its policy manifest is compatible.

    `bm25s` raises when asked for more results than the corpus has, so `top_k` is clamped to the
    manifest's node count. This only changes behavior for a corpus smaller than the candidate pool
    (`retrieval/retriever.py`'s `DEFAULT_CANDIDATE_K`); a corpus at or above that size is unaffected.
    """

    manifest = load_manifest(config)
    path = config.storage_dir / "bm25"
    if not path.exists():
        raise RuntimeError("The BM25 index is missing. Run: python scripts/ingest_documents.py")
    retriever = BM25Retriever.from_persist_dir(str(path))
    retriever.similarity_top_k = min(top_k, manifest["node_count"])
    # Not persisted by BM25Retriever: without this, queries fall back to the default pattern and
    # stop matching the one-character tokens the index was built with.
    retriever.token_pattern = BM25_TOKEN_PATTERN
    return retriever
