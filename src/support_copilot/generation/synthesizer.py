"""Label evidence, invoke LlamaIndex synthesis, and normalize abstention behavior."""

from llama_index.core import get_response_synthesizer
from llama_index.core.schema import NodeWithScore, TextNode

from support_copilot.generation.prompts import ABSTENTION_MESSAGE, EVIDENCE_PROMPT, REFINE_PROMPT


def evidence_location(metadata: dict) -> str:
    """Return a universal location with backward-compatible page handling."""

    location = metadata.get("location")
    if location not in (None, ""):
        return str(location)
    page = metadata.get("page_label", metadata.get("page"))
    return f"page:{page}" if page not in (None, "") else "unknown"


def number_evidence(evidence):
    """Create numbered node copies so answer markers have one explicit meaning.

    The prompt shows only the label, file, location, and content. Every metadata key is excluded
    from the LLM view: otherwise LlamaIndex prints all of it (hashes, IDs, ranking scores) above
    each block, where it reads as the tail of the previous source and blurs the label boundaries.
    """

    numbered = []
    for number, item in enumerate(evidence, start=1):
        metadata = dict(item.node.metadata)
        source = metadata.get("source") or metadata.get("file_name") or "unknown"
        text = "\n".join(
            [
                f"[Source {number}]",
                f"File: {source}",
                f"Location: {evidence_location(metadata)}",
                "Content:",
                item.node.get_content(),
            ]
        )
        node = TextNode(
            id_=item.node.node_id,
            text=text,
            metadata=metadata,
            excluded_llm_metadata_keys=list(metadata),
            excluded_embed_metadata_keys=list(metadata),
        )
        numbered.append(NodeWithScore(node=node, score=item.score))
    return numbered


def normalize_answer(answer: str) -> str:
    """Collapse empty or self-contradictory abstentions to one exact message."""

    cleaned = (answer or "").strip()
    if not cleaned or cleaned.startswith(ABSTENTION_MESSAGE):
        return ABSTENTION_MESSAGE
    return cleaned


def synthesize(question: str, evidence, *, synthesizer=None) -> str:
    """Generate from exactly the evidence visible to the application."""

    if not evidence:
        return ABSTENTION_MESSAGE
    synthesizer = synthesizer or get_response_synthesizer(
        text_qa_template=EVIDENCE_PROMPT,
        refine_template=REFINE_PROMPT,
        response_mode="compact",
    )
    response = synthesizer.synthesize(question, nodes=number_evidence(evidence))
    return normalize_answer(str(response))
