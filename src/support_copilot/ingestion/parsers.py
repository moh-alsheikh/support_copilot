"""Configure chunking independently from file extraction."""

from llama_index.core.node_parser import SentenceSplitter


def build_parser(config):
    """Return the sentence-aware splitter configured for this corpus."""

    return SentenceSplitter(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
