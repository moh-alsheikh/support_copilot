"""Run retrieval metrics or generated-answer evaluation.

Usage:
    python scripts/run_evaluation.py retrieval --mode hybrid --top-k 4 --run-name baseline
    python scripts/run_evaluation.py answers --mode hybrid --top-k 4 --run-name baseline
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from support_copilot.config import AppConfig, configure_llamaindex
from support_copilot.evaluation.runner import run_answers, run_retrieval


def main() -> None:
    """Execute the selected evaluation stage and print generated report paths."""

    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["retrieval", "answers"])
    parser.add_argument("--mode", choices=["semantic", "bm25", "hybrid"], default="hybrid")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--run-name", default="baseline")
    args = parser.parse_args()
    config = AppConfig.from_env()
    configure_llamaindex(config)
    top_k = args.top_k or config.top_k
    if args.stage == "retrieval":
        paths = run_retrieval(config, mode=args.mode, top_k=top_k, run_name=args.run_name)
    else:
        paths = run_answers(config, mode=args.mode, top_k=top_k, run_name=args.run_name)
    for path in paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
