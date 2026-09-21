"""Build fresh semantic and BM25 indexes from normalized records.

The full rebuild removes generated indexes first. Existing state therefore requires an explicit
`--force` flag or an interactive confirmation.

Usage:
    python scripts/ingest_documents.py
    python scripts/ingest_documents.py --force
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from support_copilot.config import AppConfig, configure_llamaindex
from support_copilot.indexing.index_manager import has_existing_indexes, manifest_path
from support_copilot.services.ingestion_service import IngestionService


def parse_args():
    """Read the explicit replacement flag used by automation and repeatable demos."""

    parser = argparse.ArgumentParser(description="Rebuild Chroma and BM25 from prepared records.")
    parser.add_argument("--force", action="store_true", help="Replace existing generated indexes.")
    return parser.parse_args()


def replacement_confirmed(config, force: bool) -> bool:
    """Protect against replacing generated indexes without noticing."""

    if force or not has_existing_indexes(config):
        return True
    answer = input("Existing Chroma/BM25 indexes will be replaced. Continue? [y/N] ").strip().lower()
    return answer in {"y", "yes"}


def main() -> None:
    """Configure embeddings, rebuild both indexes, and report their contract."""

    args = parse_args()
    config = AppConfig.from_env()
    configure_llamaindex(config)
    if not replacement_confirmed(config, args.force):
        raise SystemExit("Index rebuild cancelled; existing generated data was left unchanged.")
    _, documents, nodes = IngestionService(config).run()
    print(f"Prepared records loaded: {len(documents)}")
    print(f"Nodes stored in Chroma and BM25: {len(nodes)}")
    print(f"Index manifest: {manifest_path(config)}")
    print("A repeated --force run replaces generated indexes; it does not duplicate nodes.")


if __name__ == "__main__":
    main()
