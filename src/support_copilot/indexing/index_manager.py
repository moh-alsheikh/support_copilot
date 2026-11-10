"""Build matching Chroma and BM25 indexes with a verifiable storage manifest.

This project uses an explicit full rebuild so deleted source records cannot leave stale vectors.
The command-line script asks for confirmation before replacing existing generated indexes. A
manifest records the corpus fingerprint and indexing policy so query-time code can reject an
incompatible BM25 artifact instead of returning quietly misleading rankings.
"""

import hashlib
import json
from pathlib import Path
import shutil

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.retrievers.bm25 import BM25Retriever

from support_copilot.indexing.vector_store import get_collection, get_vector_store
from support_copilot.ingestion.pipeline import build_nodes

BM25_POLICY_VERSION = 2
# bm25s's default pattern (\b\w\w+\b) drops one-character tokens, so digits that distinguish
# otherwise-identical rows ("0 long, 7 short flashes", "Pi 5" vs "Pi 4") are never matched.
BM25_TOKEN_PATTERN = r"(?u)\b\w+\b"
MANIFEST_NAME = "index_manifest.json"


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
    return bm25_path.exists() or manifest_path(config).exists() or get_collection(config).count() > 0


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


def ingest(config):
    """Rebuild semantic and BM25 indexes from all normalized records."""

    documents, nodes = build_nodes(config)
    if not nodes:
        raise ValueError("No searchable nodes were produced. Inspect corpus preparation warnings.")

    vector_store = get_vector_store(config, reset=True)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_context)

    bm25_path = config.storage_dir / "bm25"
    if bm25_path.exists():
        shutil.rmtree(bm25_path)
    BM25Retriever.from_defaults(
        nodes=nodes, similarity_top_k=config.top_k, token_pattern=BM25_TOKEN_PATTERN
    ).persist(str(bm25_path))
    write_manifest(config, nodes)
    return index, documents, nodes


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
