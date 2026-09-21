"""Inspect ranked evidence without calling an LLM.

Usage:
    python scripts/inspect_index.py --mode hybrid --question "How do I reset my password?"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from support_copilot.config import AppConfig, configure_llamaindex
from support_copilot.retrieval.inspector import print_results
from support_copilot.retrieval.retriever import retrieve


def main() -> None:
    """Run one selected retriever and print complete evidence metadata."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["semantic", "bm25", "hybrid"], default="hybrid")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--question", default="How do I reset my password?")
    args = parser.parse_args()
    config = AppConfig.from_env()
    configure_llamaindex(config)
    print_results(retrieve(config, args.question, top_k=args.top_k, mode=args.mode))


if __name__ == "__main__":
    main()
