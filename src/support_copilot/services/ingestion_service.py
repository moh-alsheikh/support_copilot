"""Application service for the complete index-build workflow."""

from support_copilot.indexing.index_manager import ingest


class IngestionService:
    """Expose ingestion without coupling scripts or UI to indexing internals."""

    def __init__(self, config):
        """Store immutable application configuration."""

        self.config = config

    def run(self):
        """Build fresh semantic and BM25 indexes."""

        return ingest(self.config)
