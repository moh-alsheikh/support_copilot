"""Update the semantic and BM25 indexes from normalized records.

By default only new or changed documents are split and embedded, and removed documents are deleted
from the index. A full rebuild runs automatically when there's nothing to update incrementally (no
index yet, or the chunking/embedding/distance settings changed), or on request.

Usage:
    python scripts/ingest_documents.py                  # incremental update
    python scripts/ingest_documents.py --full-rebuild   # rebuild from scratch (asks to confirm)
    python scripts/ingest_documents.py --force          # rebuild from scratch, no prompt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from support_copilot.config import AppConfig, configure_llamaindex
from support_copilot.indexing.index_manager import has_existing_indexes, manifest_path
from support_copilot.services.ingestion_service import IngestionService


def parse_args():
    """Read the rebuild flags used by automation and repeatable demos."""

    parser = argparse.ArgumentParser(description="Update Chroma and BM25 from prepared records.")
    parser.add_argument("--full-rebuild", action="store_true", help="Discard existing indexes and rebuild.")
    parser.add_argument("--force", action="store_true", help="Full rebuild without the confirmation prompt.")
    return parser.parse_args()


def rebuild_confirmed(config) -> bool:
    """Protect against discarding generated indexes without noticing."""

    if not has_existing_indexes(config):
        return True
    answer = input("Existing Chroma/BM25 indexes will be discarded and rebuilt. Continue? [y/N] ")
    return answer.strip().lower() in {"y", "yes"}


def main() -> None:
    """Configure embeddings, update both indexes, and report what changed."""

    args = parse_args()
    config = AppConfig.from_env()
    configure_llamaindex(config)
    full_rebuild = args.force or args.full_rebuild
    if args.full_rebuild and not args.force and not rebuild_confirmed(config):
        raise SystemExit("Rebuild cancelled; existing generated data was left unchanged.")
    report = IngestionService(config).run(full_rebuild=full_rebuild)
    mode = report.mode if not report.reason else f"{report.mode} ({report.reason})"
    print(f"Mode: {mode}")
    print(f"Documents: {report.documents} (added {report.added}, changed {report.changed}, "
          f"removed {report.removed}, unchanged {report.unchanged})")
    print(f"Nodes embedded this run: {report.nodes_embedded}")
    print(f"Nodes in Chroma and BM25: {report.nodes_total}")
    print(f"Index manifest: {manifest_path(config)}")


if __name__ == "__main__":
    main()
