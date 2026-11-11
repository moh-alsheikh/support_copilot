"""Claim-level answer checks, built from real failure patterns seen in model comparisons."""

from llama_index.core.schema import NodeWithScore, TextNode

from support_copilot.evaluation.evaluators import _values, answer_checks, claim_checks, summarize_answers
from support_copilot.generation.prompts import ABSTENTION_MESSAGE

USB_NOTE = (
    "NOTE: The Raspberry Pi 5 provides 1.6A of power to downstream USB peripherals when connected to a "
    "power supply capable of 5A at +5V (25W). When connected to any other compatible power supply, the "
    "Raspberry Pi 5 restricts downstream USB devices to 600mA of power."
)
PSU_BRIEF = "The Raspberry Pi 27W USB-C Power Supply delivers a maximum of 5.1V, 5A and supports USB PD."


def _case(expected: str, question_type: str = "supported") -> dict:
    return {
        "expected_answer": expected,
        "question_type": question_type,
        "expected_sources": ["power-supplies.md"] if question_type == "supported" else [],
        "expected_locations": ["lines:114-114"] if question_type == "supported" else [],
    }


def _evidence(*texts: str) -> tuple[list, list[dict]]:
    nodes, citations = [], []
    for number, text in enumerate(texts, start=1):
        location = "lines:114-114" if text is USB_NOTE else f"page:{number}"
        source = "power-supplies.md" if text is USB_NOTE else "27w-brief.pdf"
        nodes.append(NodeWithScore(node=TextNode(text=text, metadata={"source": source, "location": location}), score=1.0))
        citations.append({"number": number, "source": source, "location": location})
    return nodes, citations


def test_correct_cited_answer_passes_every_check() -> None:
    nodes, citations = _evidence(USB_NOTE)
    answer = "Without a 5A supply, the Raspberry Pi 5 limits downstream USB devices to 600mA [Source 1]."
    row = answer_checks(answer, _case("600"), citations, nodes)
    assert (row["unsupported_values"], row["weakly_supported_claims"], row["needs_review"]) == ([], [], False)


def test_inference_stating_a_value_absent_from_the_source_is_flagged_for_review() -> None:
    # Correct, but "3A" is inferred, never stated: a lexical check can't tell this from an invented
    # value, so it routes the answer to human review rather than passing or failing it.
    nodes, citations = _evidence(USB_NOTE)
    row = answer_checks("With a 3A supply, USB devices get 600mA [Source 1].", _case("600"), citations, nodes)
    assert (row["answer_contains_expected"], row["unsupported_values"], row["needs_review"]) == (True, ["3a"], True)


def test_invented_value_behind_a_valid_citation_is_flagged() -> None:
    # granite4:micro's real Q003 answer: a real source, a claim it doesn't support.
    nodes, citations = _evidence(USB_NOTE)
    answer = "When powered by a 3A supply, the Raspberry Pi 5 can provide up to 1.6A to USB peripherals [Source 1]."
    row = answer_checks(answer, _case("600"), citations, nodes)
    assert row["invalid_citation_markers"] == []
    assert row["unsupported_values"] == ["3a"]
    assert row["answer_missing_expected"] is True
    assert row["needs_review"] is True


def test_claim_pinned_to_the_wrong_source_is_flagged() -> None:
    claims = claim_checks(
        "The Raspberry Pi 5 restricts downstream USB devices to 600mA [Source 2].", [USB_NOTE, PSU_BRIEF]
    )
    assert claims["unsupported_values"] == ["600ma"]


def test_expected_answer_matching_respects_word_boundaries() -> None:
    nodes, citations = _evidence(USB_NOTE)

    def contains(answer: str, expected: str) -> bool:
        return answer_checks(answer, _case(expected), citations, nodes)["answer_contains_expected"]

    assert contains("I know the answer [Source 1].", "No") is False
    assert contains("It is not supported [Source 1].", "No") is False
    assert contains("No, none do [Source 1].", "No") is True
    assert contains("Limited to 600mA [Source 1].", "600") is True
    assert contains("Limited to 6000mA [Source 1].", "600") is False
    assert contains("The power banks can overheat and catch fire.", "overheat") is True
    assert contains("They were recalled for overheating.", "overheat") is True


def test_answering_a_near_miss_needs_review_and_a_correct_refusal_does_not() -> None:
    nodes, citations = _evidence(PSU_BRIEF)
    answered = answer_checks("The maximum output is 5.1V, 5A [Source 1].", _case("", "unsupported"), citations, nodes)
    refused = answer_checks(ABSTENTION_MESSAGE, _case("", "unsupported"), citations, nodes)
    assert (answered["false_answer"], answered["needs_review"]) == (True, True)
    assert (refused["abstention_correct"], refused["needs_review"]) == (True, False)


def test_values_ignore_words_markers_and_location_references() -> None:
    assert _values("When a Raspberry Pi 5 is turned off it uses 1 W to 1.4 W") == {"5", "1w", "1.4w"}
    assert _values("stated on lines 161-161 and page 1") == set()
    assert claim_checks("Fan speed is [Source 1]8000 RPM[/Source 1].", ["Maximum fan speed: 8000 RPM"])["unsupported_values"] == []


def test_summary_lists_questions_per_signal() -> None:
    nodes, citations = _evidence(USB_NOTE)
    rows = [
        {"question_id": "Q1", "question_type": "supported", **answer_checks("600mA [Source 1].", _case("600"), citations, nodes)},
        {"question_id": "Q2", "question_type": "supported", **answer_checks(ABSTENTION_MESSAGE, _case("600"), citations, nodes)},
    ]
    summary = summarize_answers(rows)
    assert (summary["answered_supported"], summary["false_abstentions"], summary["needs_review"]) == (1, ["Q2"], ["Q2"])
