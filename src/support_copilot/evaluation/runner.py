"""Run named retrieval or answer experiments and save inspectable reports."""

import csv
import json
import time

from support_copilot.evaluation.dataset import load_cases
from support_copilot.evaluation.evaluators import (
    answer_checks,
    evaluate_retrieval,
    manual_rubric,
    summarize,
    summarize_answers,
)
from support_copilot.retrieval.retriever import retrieve
from support_copilot.services.query_service import QueryService


def _serializable(row: dict) -> dict:
    """Convert nested values into readable JSON strings for CSV."""

    return {
        key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict, tuple)) else value
        for key, value in row.items()
    }


def write_csv(rows: list[dict], path) -> None:
    """Write non-empty report rows with a stable header."""

    if not rows:
        raise ValueError("The evaluation dataset contains no cases.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(_serializable(row) for row in rows)


def run_retrieval(config, *, mode: str, top_k: int, run_name: str):
    """Evaluate one retriever and return detail/summary report paths."""

    cases = load_cases(config.evaluation_dataset_path)
    rows = []
    for case in cases:
        started = time.perf_counter()
        results = retrieve(config, case["question"], top_k=top_k, mode=mode)
        rows.append(
            {
                "question_id": case["question_id"],
                "question": case["question"],
                "retriever": mode,
                "top_k": top_k,
                "latency_seconds": round(time.perf_counter() - started, 4),
                **evaluate_retrieval(results, case, top_k=top_k),
            }
        )
    output_dir = config.reports_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / f"{mode}_retrieval_details.csv"
    summary_path = output_dir / f"{mode}_retrieval_summary.json"
    write_csv(rows, details_path)
    summary_path.write_text(json.dumps(summarize(rows), indent=2), encoding="utf-8")
    return details_path, summary_path


def run_answers(config, *, mode: str, top_k: int, run_name: str):
    """Generate answers and create automatic observations plus a blank human rubric."""

    cases = load_cases(config.evaluation_dataset_path)
    service = QueryService(config)
    rows = []
    for case in cases:
        started = time.perf_counter()
        result = service.answer(case["question"], top_k=top_k, mode=mode)
        rows.append(
            {
                "question_id": case["question_id"],
                "question": case["question"],
                "question_type": case["question_type"],
                "answer": result.answer,
                "model_answer": result.model_answer,
                "citations": [citation.__dict__ for citation in result.citations],
                "provider": config.llm_provider,
                "model": config.llm_model if config.llm_provider == "ollama" else config.anthropic_model,
                "retriever": mode,
                "top_k": top_k,
                "latency_seconds": round(time.perf_counter() - started, 4),
                **answer_checks(
                    result.answer,
                    case,
                    [citation.__dict__ for citation in result.citations],
                    result.evidence,
                    result.claims,
                ),
            }
        )
    output_dir = config.reports_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    answers_path = output_dir / "generated_answers.csv"
    rubric_path = output_dir / "manual_answer_evaluation.csv"
    summary_path = output_dir / "answer_summary.json"
    write_csv(rows, answers_path)
    write_csv(manual_rubric(rows), rubric_path)
    summary_path.write_text(json.dumps(summarize_answers(rows), indent=2), encoding="utf-8")
    return answers_path, rubric_path, summary_path


def run(config):
    """Backward-compatible default used by older notebooks and scripts."""

    return run_retrieval(config, mode="hybrid", top_k=config.top_k, run_name="baseline")[0]
