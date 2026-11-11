"""Pair-aware retrieval metrics and deterministic answer/citation checks.

The program checks facts it can know: retrieved identities, source-label ranges, exact abstention
behavior, and whether each cited sentence is backed by the text of the sources it cites (values it
states must appear there, and most of its words should). Since `[Source N]` markers are attached by
code (`generation/attribution.py`) using the same rules, cited sentences pass those checks by
construction; the signal moves to `unverified_claims`, the sentences code could not support. Those claim checks are lexical, so they
catch invented values and claims pinned to the wrong source, but not a sentence that reuses the
source's words to say something it doesn't mean. Human reviewers remain authoritative for
correctness and faithfulness; `needs_review` tells them where to look first.
"""

import re

from support_copilot.generation.attribution import (
    MARKER,
    SUPPORT_THRESHOLD,
    citation_markers,
    sentences,
    support_text,
    unsupported_values,
    values as _values,
    word_support,
)
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


def _contains_phrase(text: str, phrase: str) -> bool:
    """Match the phrase at a word start, so an expected "no" doesn't match inside "know".

    After the phrase, numbers may carry a unit ("600" matches "600mA") and longer words may carry
    a suffix ("overheat" matches "overheating"); short words like "no" must end at a boundary, or
    they would match "not" and "none".
    """

    phrase = _normalized_words(phrase)
    if not phrase:
        return False
    if phrase[-1].isdigit():
        tail = r"(?![0-9])"
    elif len(phrase) <= 3:
        tail = r"(?![a-z0-9])"
    else:
        tail = ""
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}{tail}", _normalized_words(text)) is not None


def _normalized_words(text: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", text.lower()).split())


def claim_checks(answer: str, evidence_texts: list[str]) -> dict:
    """Check each sentence that cites `[Source N]` against the text of exactly those sources.

    `evidence_texts[n - 1]` is the text shown to the model as `[Source n]`. Sentences citing only
    nonexistent sources are skipped here (they are already counted as invalid markers).
    """

    missing_values: list[str] = []
    weak_claims: list[str] = []
    uncited_values: list[str] = []
    checked = 0
    for sentence in sentences(answer):
        markers = citation_markers(sentence)
        claim = MARKER.sub(" ", sentence)
        if not markers:
            uncited_values.extend(sorted(_values(claim)))
            continue
        cited = " ".join(evidence_texts[n - 1] for n in sorted(markers) if 1 <= n <= len(evidence_texts))
        if not cited:
            continue
        checked += 1
        missing_values.extend(unsupported_values(claim, cited))
        overlap = word_support(claim, cited)
        if overlap is not None and overlap < SUPPORT_THRESHOLD:
            weak_claims.append(claim.strip()[:160])
    return {
        "claims_checked": checked,
        "unsupported_values": missing_values,
        "weakly_supported_claims": weak_claims,
        "uncited_values": uncited_values,
    }


def answer_checks(answer: str, case: dict, citations: list[dict], evidence, claims=None) -> dict:
    """Check abstention, label validity, expected evidence, and claim support deterministically.

    `claims` are the per-sentence attributions from `generation/attribution.py`, when the answer
    went through it. They add `unverified_claims` (sentences no source supports, shown to the user
    as `[Unverified]`) and `model_citation_fixes` (sentences where code's citation differs from the
    model's own marker, a diagnostic of how well the model cites).
    """

    abstained = ABSTENTION_MESSAGE.lower() in answer.lower()
    unsupported = case["question_type"] == "unsupported"
    markers = citation_markers(answer)
    citations_by_number = {int(item["number"]): (item["source"], item["location"]) for item in citations}
    invalid_markers = sorted(markers.difference(citations_by_number))
    cited_identities = {citations_by_number[number] for number in markers if number in citations_by_number}
    retrieved_identities = {evidence_identity(item) for item in evidence}
    expected = expected_evidence(case)
    contains_expected = _contains_phrase(answer, case["expected_answer"])
    claim_results = claim_checks(answer, [support_text(item.node) for item in evidence]) if not abstained else claim_checks("", [])
    row = {
        "answer_contains_expected": contains_expected,
        "abstained": abstained,
        "abstention_correct": abstained if unsupported else not abstained,
        "false_answer": unsupported and not abstained,
        "false_abstention": not unsupported and abstained,
        "answer_missing_expected": not unsupported and not abstained and not contains_expected,
        "citation_markers": sorted(markers),
        "invalid_citation_markers": invalid_markers,
        "citations_not_retrieved": sorted(cited_identities.difference(retrieved_identities)),
        "expected_evidence_cited": None if unsupported else bool(cited_identities.intersection(expected)),
        **claim_results,
        "unverified_claims": [claim.text[:160] for claim in claims or [] if claim.status == "unverified"],
        "model_citation_fixes": [
            f"{_labels(claim.model_sources)} -> {_labels(claim.sources)}: {claim.text[:100]}"
            for claim in claims or []
            if claim.status != "not_a_claim" and set(claim.model_sources) != set(claim.sources)
        ],
    }
    row["needs_review"] = needs_review(row)
    return row


def _labels(numbers) -> str:
    return "".join(f"[Source {n}]" for n in numbers) or "none"


def needs_review(row: dict) -> bool:
    """True when any automatic check suggests the answer may be wrong or ungrounded."""

    return bool(
        row["false_answer"]
        or row["false_abstention"]
        or row["answer_missing_expected"]
        or row["invalid_citation_markers"]
        or row["citations_not_retrieved"]
        or row["unsupported_values"]
        or row["weakly_supported_claims"]
        or row.get("unverified_claims")
        or row["expected_evidence_cited"] is False
    )


def summarize_answers(rows: list[dict]) -> dict:
    """Count each failure signal across an answer run, plus which questions raised it."""

    def ids(predicate) -> list[str]:
        return [row["question_id"] for row in rows if predicate(row)]

    supported = [row for row in rows if row["question_type"] != "unsupported"]
    return {
        "questions": len(rows),
        "answered_supported": sum(not row["abstained"] for row in supported),
        "supported_questions": len(supported),
        "false_answers": ids(lambda r: r["false_answer"]),
        "false_abstentions": ids(lambda r: r["false_abstention"]),
        "answers_missing_expected": ids(lambda r: r["answer_missing_expected"]),
        "invalid_citation_markers": ids(lambda r: r["invalid_citation_markers"]),
        "unsupported_values": ids(lambda r: r["unsupported_values"]),
        "weakly_supported_claims": ids(lambda r: r["weakly_supported_claims"]),
        "unverified_claims": ids(lambda r: r.get("unverified_claims")),
        "model_citation_fixes": ids(lambda r: r.get("model_citation_fixes")),
        "expected_evidence_not_cited": ids(lambda r: r["expected_evidence_cited"] is False and not r["abstained"]),
        "needs_review": ids(lambda r: r["needs_review"]),
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
            "auto_needs_review": row.get("needs_review", ""),
            "reviewer_notes": "",
        }
        for row in rows
    ]
