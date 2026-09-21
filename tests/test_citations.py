from support_copilot.generation.citations import citations_from_nodes
from support_copilot.generation.prompts import ABSTENTION_MESSAGE, EVIDENCE_PROMPT, REFINE_PROMPT
from support_copilot.generation.synthesizer import normalize_answer, number_evidence

from llama_index.core.schema import NodeWithScore, TextNode


class DummyNode:
    node_id = "node-x"
    metadata = {"source": "reports/manual.pdf", "location": "page:7"}


class DummyScored:
    node = DummyNode()


def test_citations_from_nodes():
    citation = citations_from_nodes([DummyScored()])[0]
    assert citation.source == "reports/manual.pdf"
    assert citation.location == "page:7"
    assert citation.chunk_id == "node-x"


def test_prompt_and_refine_prompt_share_grounding_contract() -> None:
    prompt = EVIDENCE_PROMPT.format(context_str="[Source 1] evidence", query_str="Question?")
    refine = REFINE_PROMPT.format(
        query_str="Question?", existing_answer="Answer", context_msg="[Source 2] evidence"
    )
    for rendered in (prompt, refine):
        assert ABSTENTION_MESSAGE in rendered
        assert "outside knowledge" in rendered


def test_duplicate_locations_do_not_break_evidence_label_alignment() -> None:
    first = TextNode(id_="one", text="A", metadata={"source": "same.txt", "location": "line:1"})
    second = TextNode(id_="two", text="B", metadata={"source": "same.txt", "location": "line:1"})
    evidence = [NodeWithScore(node=first, score=1.0), NodeWithScore(node=second, score=0.9)]
    assert number_evidence(evidence)[1].node.get_content().startswith("[Source 2]")
    assert [item.number for item in citations_from_nodes(evidence)] == [1, 2]


def test_contradictory_abstention_is_normalized() -> None:
    raw = ABSTENTION_MESSAGE + " However, an unsupported claim follows."
    assert normalize_answer(raw) == ABSTENTION_MESSAGE
