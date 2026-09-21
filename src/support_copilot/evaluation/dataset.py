"""Load and validate the evaluation contract."""

import json


REQUIRED_FIELDS = {
    "question_id",
    "question",
    "expected_answer",
    "expected_sources",
    "expected_locations",
    "question_type",
}


def load_cases(path) -> list[dict]:
    """Read JSON cases and reject incomplete ground truth early."""

    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("Evaluation data must be a top-level JSON array.")
    for number, case in enumerate(cases, start=1):
        missing = REQUIRED_FIELDS.difference(case)
        if missing:
            raise ValueError(f"Case {number} is missing: {', '.join(sorted(missing))}")
        if not isinstance(case["expected_sources"], list) or not isinstance(
            case["expected_locations"], list
        ):
            raise ValueError(f"Case {number}: expected sources and locations must be JSON lists.")
        if len(case["expected_sources"]) != len(case["expected_locations"]):
            raise ValueError(f"Case {number}: expected sources and locations must align by position.")
        if case["question_type"] == "unsupported" and case["expected_sources"]:
            raise ValueError(f"Case {number}: unsupported questions cannot contain expected evidence.")
    return cases
