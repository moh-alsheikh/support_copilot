"""Turn prepared Documents into explicit LlamaIndex nodes."""

from .document_adapter import load_documents
from .parsers import build_parser


def build_nodes(config):
    """Load normalized records and split them while retaining source/location metadata."""

    documents = load_documents(config)
    nodes = build_parser(config).get_nodes_from_documents(documents)
    return documents, nodes
