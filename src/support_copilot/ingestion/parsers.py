"""Configure chunking independently from file extraction."""

from llama_index.core.node_parser import SentenceSplitter


def chunk_id(index: int, document) -> str:
    """Derive a chunk ID from its document's stable ID and its position in that document.

    LlamaIndex's default assigns a random UUID per chunk, so an unchanged corpus produced new IDs
    (and a new corpus fingerprint) on every build, and incremental upserts couldn't line up chunks.
    """

    return f"{document.node_id}:{index}"


def build_parser(config):
    """Return the sentence-aware splitter configured for this corpus."""

    return SentenceSplitter(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap, id_func=chunk_id)
