"""Pair-aware retrieval metrics and deterministic answer/citation checks.

The program checks facts it can know: retrieved identities, source-label ranges, and exact
abstention behavior. Human reviewers remain authoritative for correctness and faithfulness.
"""

import re

from support_copilot.generation.prompts import ABSTENTION_MESSAGE


def evidence_identity(item) -> tuple[str, str]:
    """Return the universal source/location identity of a scored node."""

    metadata = item.node.metadata
    return (
        str(metadata.get("source") or metadata.get("file_name") or "unknown"),
        str(metadata.get("location") or "unknown"),
    )


def expected_evidence(case: dict) -> set[tuple[str, str]]:
    """Pair expected sources and locations exactly as validated by the dataset loader."""

    return set(zip(case["expected_sources"], case["expected_locations"], strict=True))


def evaluate_retrieval(results, case: dict, *, top_k: int) -> dict:
    """Calculate Hit@K, Recall@K, context precision, and reciprocal rank."""

    selected = results[:top_k]
    identities = [evidence_identity(item) for item in selected]
    evidence_rows = [
        {
            "rank": rank,
            "source": evidence_identity(item)[0],
            "location": evidence_identity(item)[1],
            "node_id": item.node.node_id,
            "score": item.score,
            "semantic_rank": item.node.metadata.get("semantic_rank"),
            "bm25_rank": item.node.metadata.get("bm25_rank"),
            "text": item.node.get_content(),
        }
        for rank, item in enumerate(selected, start=1)
    ]
    if case["question_type"] == "unsupported":
        return {"unsupported": True, "retrieved_evidence": evidence_rows}

    expected = expected_evidence(case)
    relevance = [identity in expected for identity in identities]
    found = len(set(identities).intersection(expected))
    first_rank = next((rank for rank, value in enumerate(relevance, start=1) if value), None)
    return {
        "unsupported": False,
        "retrieved_evidence": evidence_rows,
        "hit_at_k": float(any(relevance)),
        "recall_at_k": found / len(expected) if expected else 0.0,
        "context_precision": sum(relevance) / len(relevance) if relevance else 0.0,
        "reciprocal_rank": 1.0 / first_rank if first_rank else 0.0,
    }


def summarize(rows: list[dict]) -> dict:
    """Average supported cases and count unsupported cases separately."""

    supported = [row for row in rows if not row["unsupported"]]

    def average(field: str) -> float:
        """Average one metric over supported questions only."""

        return sum(row[field] for row in supported) / len(supported) if supported else 0.0

    return {
        "supported_questions": len(supported),
        "unsupported_questions": len(rows) - len(supported),
        "hit_at_k": average("hit_at_k"),
        "recall_at_k": average("recall_at_k"),
        "context_precision": average("context_precision"),
        "mrr": average("reciprocal_rank"),
    }


def citation_markers(answer: str) -> set[int]:
    """Return explicit `[Source N]` markers from generated prose."""

    return {int(value) for value in re.findall(r"\[Source\s+(\d+)\]", answer, re.IGNORECASE)}


def answer_checks(answer: str, case: dict, citations: list[dict], evidence) -> dict:
    """Check abstention, label validity, and expected cited evidence deterministically."""

    normalized_answer = " ".join(re.sub(r"[^\w]+", " ", answer.lower()).split())
    normalized_expected = " ".join(re.sub(r"[^\w]+", " ", case["expected_answer"].lower()).split())
    abstained = ABSTENTION_MESSAGE.lower() in answer.lower()
    unsupported = case["question_type"] == "unsupported"
    markers = citation_markers(answer)
    citations_by_number = {int(item["number"]): (item["source"], item["location"]) for item in citations}
    invalid_markers = sorted(markers.difference(citations_by_number))
    cited_identities = {citations_by_number[number] for number in markers if number in citations_by_number}
    retrieved_identities = {evidence_identity(item) for item in evidence}
    expected = expected_evidence(case)
    return {
        "answer_contains_expected": bool(normalized_expected) and normalized_expected in normalized_answer,
        "abstained": abstained,
        "abstention_correct": abstained if unsupported else not abstained,
        "false_answer": unsupported and not abstained,
        "false_abstention": not unsupported and abstained,
        "citation_markers": sorted(markers),
        "invalid_citation_markers": invalid_markers,
        "citations_not_retrieved": sorted(cited_identities.difference(retrieved_identities)),
        "expected_evidence_cited": None if unsupported else bool(cited_identities.intersection(expected)),
    }


def manual_rubric(rows: list[dict]) -> list[dict]:
    """Prepare authoritative 0/1/2 human scoring columns."""

    return [
        {
            "question_id": row["question_id"],
            "question": row["question"],
            "answer": row.get("answer", ""),
            "citations": row.get("citations", []),
            "correctness_0_2": "",
            "faithfulness_0_2": "",
            "completeness_0_2": "",
            "citation_accuracy_0_2": "",
            "abstention_correct": row.get("abstention_correct", ""),
            "reviewer_notes": "",
        }
        for row in rows
    ]
