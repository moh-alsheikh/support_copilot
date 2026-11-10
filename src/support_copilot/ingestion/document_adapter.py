"""Convert normalized records into stable LlamaIndex Document objects.

Format-specific extraction happens before this adapter. LlamaIndex therefore receives one common
contract regardless of whether evidence came from a PDF page, Word table, CSV row, or JSON path.
"""

import hashlib

from llama_index.core import Document

from .processed_records import load_records

# Only these metadata fields join the text that is embedded and BM25-indexed (both use LlamaIndex's
# EMBED view). Everything else — hashes, IDs, encoding, location — stays in metadata for citations
# and evaluation but would otherwise dominate short chunks (a one-line table row carried nine
# metadata lines, including a 64-character content hash) and turn digits like line numbers into
# BM25 matches.
EMBEDDED_METADATA_KEYS = ("title", "topic")


def stable_record_id(record: dict) -> str:
    """Derive a repeatable ID from the document ID, source, location, and text."""

    metadata = record["metadata"]
    identity = "|".join(
        [
            str(record.get("doc_id", metadata.get("doc_id", ""))),
            str(metadata.get("source", "")),
            str(metadata.get("location", "")),
            record["text"],
        ]
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def load_documents(config):
    """Read prepared JSONL records and preserve universal citation metadata."""

    if not config.processed_records_path.exists():
        raise FileNotFoundError(
            f"Missing {config.processed_records_path}. Run: python prepare_corpus.py"
        )
    documents = []
    for record in load_records(config.processed_records_path):
        metadata = dict(record["metadata"])
        metadata.setdefault("doc_id", record.get("doc_id", ""))
        hidden = [key for key in metadata if key not in EMBEDDED_METADATA_KEYS]
        documents.append(
            Document(
                id_=stable_record_id(record),
                text=record["text"],
                metadata=metadata,
                excluded_embed_metadata_keys=hidden,
                excluded_llm_metadata_keys=hidden,
            )
        )
    return documents
