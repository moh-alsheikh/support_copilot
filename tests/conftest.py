"""Shared fixtures keep tests offline and isolated from real corpus data."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from support_copilot.config import AppConfig


@pytest.fixture
def make_config(tmp_path):
    """Return a factory for complete temporary AppConfig instances."""

    def factory() -> AppConfig:
        return AppConfig(
            root_dir=tmp_path,
            raw_data_dir=tmp_path / "data" / "raw",
            processed_records_path=tmp_path / "data" / "processed" / "document_records.jsonl",
            document_register_path=tmp_path / "data" / "metadata" / "document_register.csv",
            readiness_path=tmp_path / "data" / "metadata" / "project_readiness.csv",
            evaluation_dataset_path=tmp_path / "evaluation" / "evaluation_questions.json",
            reports_dir=tmp_path / "reports",
            storage_dir=tmp_path / "storage",
            llm_provider="ollama",
            llm_model="mock",
            embedding_model="mock",
        )

    return factory
