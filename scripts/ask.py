"""Ask one question end to end: retrieve evidence, generate a grounded answer, list its sources.

Usage:
    python scripts/ask.py --question "Which power supply is recommended for the Raspberry Pi 5?"
    python scripts/ask.py --mode hybrid --top-k 4 --question "..."
"""

import argparse
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from support_copilot.config import AppConfig, configure_llamaindex
from support_copilot.evaluation.evaluators import citation_markers
from support_copilot.services.query_service import QueryService


def main() -> None:
    """Answer one question with the configured LLM and print the answer and its sources."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--mode", choices=["semantic", "bm25", "hybrid"], default="hybrid")
    parser.add_argument("--top-k", type=int)
    args = parser.parse_args()
    config = AppConfig.from_env()
    configure_llamaindex(config)

    started = time.perf_counter()
    result = QueryService(config).answer(args.question, top_k=args.top_k, mode=args.mode)
    elapsed = time.perf_counter() - started

    cited = citation_markers(result.answer)
    print(f"ANSWER ({config.llm_model}, {elapsed:.1f}s)\n{result.answer}\n")
    print("SOURCES (* = cited in the answer)")
    for citation in result.citations:
        mark = "*" if citation.number in cited else " "
        print(f" {mark} [Source {citation.number}] {citation.source} | {citation.location}")


if __name__ == "__main__":
    main()
