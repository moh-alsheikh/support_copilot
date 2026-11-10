"""Application service for the index-build workflow."""

from support_copilot.indexing.index_manager import IngestionReport, ingest


class IngestionService:
    """Expose ingestion without coupling scripts or UI to indexing internals."""

    def __init__(self, config):
        """Store immutable application configuration."""

        self.config = config

    def run(self, *, full_rebuild: bool = False) -> IngestionReport:
        """Update the indexes incrementally, or rebuild them from scratch when asked."""

        return ingest(self.config, full_rebuild=full_rebuild)
