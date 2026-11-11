"""Code-attached citations: the model's markers are hints; code decides what each sentence cites."""

from llama_index.core.schema import MetadataMode, NodeWithScore, TextNode

from support_copilot.evaluation.evaluators import answer_checks
from support_copilot.generation.attribution import attribute, support_text
from support_copilot.generation.synthesizer import number_evidence

SPEC = "MTBF Ground Benign: 93 800 hours\nOperating temperature: 0°C to 70°C\nProduction lifetime: until at least January 2036"
FCC = "This device complies with part 15 of the FCC Rules and must be installed 20cm from all persons."
RECALL = "The power banks can overheat, posing a fire hazard. Consumers should stop using them."


def test_uncited_correct_answer_gets_a_citation_from_code() -> None:
    # The real Q007 answer: correct, but the model wrote no marker.
    answer, claims = attribute("1. Operating temperature:  0°C to 70°C", [FCC, SPEC])
    assert answer == "1. Operating temperature: 0°C to 70°C [Source 2]"
    assert (claims[0].status, claims[0].sources, claims[0].model_sources) == ("cited", [2], [])


def test_a_model_marker_is_kept_only_when_code_confirms_it() -> None:
    answer, _ = attribute("The power banks can overheat and cause a fire [Source 2].", [FCC, RECALL])
    assert answer == "The power banks can overheat and cause a fire [Source 2]."
    answer, claims = attribute("The power banks can overheat and cause a fire [Source 1].", [FCC, RECALL])
    assert answer == "The power banks can overheat and cause a fire [Source 2]."
    assert (claims[0].model_sources, claims[0].sources) == ([1], [2])


def test_an_invented_value_is_labelled_unverified_not_cited() -> None:
    answer, claims = attribute("The operating temperature is 0°C to 85°C [Source 1].", [SPEC])
    assert answer == "The operating temperature is 0°C to 85°C [Unverified]."
    assert claims[0].status == "unverified"


def test_filler_sentences_are_unverified_and_connectives_left_alone() -> None:
    text = "Yes.\nThe power banks can overheat [Source 1]. Additionally, [Source 2] also provides similar information."
    answer, claims = attribute(text, [RECALL, FCC])
    assert answer == (
        "Yes.\nThe power banks can overheat [Source 1]. Additionally, also provides similar information [Unverified]."
    )
    assert [claim.status for claim in claims] == ["not_a_claim", "cited", "unverified"]


def test_a_sentence_combining_two_sources_the_model_named_is_cited_to_both() -> None:
    answer, _ = attribute(
        "Operating temperature runs to 70°C and installation needs 20cm from persons [Source 1][Source 2].",
        [SPEC, FCC],
    )
    assert answer.endswith("[Source 1][Source 2].")


def test_unverified_claims_route_the_answer_to_review() -> None:
    nodes = [NodeWithScore(node=TextNode(text=SPEC, metadata={"source": "brief.pdf", "location": "page:3"}), score=1.0)]
    citations = [{"number": 1, "source": "brief.pdf", "location": "page:3"}]
    case = {"expected_answer": "70", "question_type": "supported", "expected_sources": ["brief.pdf"], "expected_locations": ["page:3"]}
    answer, claims = attribute("It runs from 0°C to 70°C. It is rated for 5 years of outdoor use.", [SPEC])
    row = answer_checks(answer, case, citations, nodes, claims)
    assert row["unverified_claims"] == ["It is rated for 5 years of outdoor use."]
    assert row["model_citation_fixes"] == ["none -> [Source 1]: It runs from 0°C to 70°C."]
    assert row["needs_review"] is True


def test_prompt_evidence_shows_no_metadata_dump() -> None:
    metadata = {"source": "brief.pdf", "location": "page:3", "content_hash": "abc123", "rrf_score": 0.03}
    numbered = number_evidence([NodeWithScore(node=TextNode(text=SPEC, metadata=metadata), score=1.0)])
    prompt_text = numbered[0].node.get_content(metadata_mode=MetadataMode.LLM)
    assert prompt_text.startswith("[Source 1]\nFile: brief.pdf\nLocation: page:3\nContent:\n")
    assert "abc123" not in prompt_text and "rrf_score" not in prompt_text


def test_phrases_that_only_carried_model_markers_are_removed() -> None:
    answer, _ = attribute(
        "[Source 2]\n\nAccording to [Source 2], page 1, operating temperature is 0°C to 70°C.\n\nEvidence: [Source 2]",
        [FCC, SPEC],
    )
    assert answer == "Operating temperature is 0°C to 70°C [Source 2]."
    answer, _ = attribute("[Source 1] states that the power banks can overheat.", [RECALL])
    assert answer == "The power banks can overheat [Source 1]."


def test_marker_subjects_titles_and_list_endings_render_cleanly() -> None:
    answer, _ = attribute("[Source 1] recommends the 27 W USB-C Power Supply.", ["We recommend the 27 W USB-C Power Supply."])
    assert answer == "The source recommends the 27 W USB-C Power Supply [Source 1]."
    answer, _ = attribute("* 5.1V, 5A\n**Product defect refunds:**", ["Outputs: 5.1V, 5A", "Refunds for product defect"])
    assert answer == "* 5.1V, 5A [Source 1]\n**Product defect refunds:** [Source 2]"


def test_the_source_title_counts_as_support() -> None:
    node = TextNode(text="o GPIO 14 (TXD)\nGround o", metadata={"title": "Gpio Pinout Diagram"})
    claim = "According to the GPIO pinout diagram, GPIO 14 carries the UART TXD signal."
    assert attribute(claim, [node.get_content()])[1][0].status == "unverified"
    assert attribute(claim, [support_text(node)])[1][0].sources == [1]


def test_marker_lists_and_abstention_sentences_survive_cleanup() -> None:
    answer, _ = attribute(
        "[Source 1] and [Source 2]\n\nThe power banks can overheat.\n\nI don't know based on the provided documents.",
        [RECALL, FCC],
    )
    assert answer == "The power banks can overheat [Source 1].\n\nI don't know based on the provided documents."
