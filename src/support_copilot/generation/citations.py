"""Create evidence-label-aligned citations from retrieval metadata, never model prose."""

from support_copilot.generation.synthesizer import evidence_location
from support_copilot.models import Citation


def citations_from_nodes(nodes):
    """Create one citation per evidence block without breaking source-label numbering."""

    citations = []
    for number, item in enumerate(nodes, start=1):
        metadata = item.node.metadata
        source = str(metadata.get("source") or metadata.get("file_name") or "unknown")
        citations.append(
            Citation(
                number=number,
                source=source,
                location=evidence_location(metadata),
                chunk_id=item.node.node_id,
            )
        )
    return citations
