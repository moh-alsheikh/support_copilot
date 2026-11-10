"""Incremental indexing: only new or changed documents are embedded; removed ones are deleted."""

from dataclasses import replace
import shutil

import pytest
from llama_index.core import Settings
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.vector_stores.utils import metadata_dict_to_node

from support_copilot.indexing.index_manager import ingest, load_bm25, load_manifest, pipeline_dir
from support_copilot.indexing.vector_store import get_collection
from support_copilot.ingestion.processed_records import write_records


def _record(text: str, location: str) -> dict:
    return {
        "doc_id": "guide",
        "text": text,
        "metadata": {"source": "guide.md", "location": location, "document_type": "markdown", "extraction_method": "test"},
    }


RECORDS = [
    _record("The Active Cooler fan reaches 8000 RPM.", "lines:1-1"),
    _record("The 27W power supply delivers 5.1V at 5A.", "lines:3-3"),
    _record("Low-voltage detection triggers below 4.63V.", "lines:5-5"),
]


@pytest.fixture
def config(make_config):
    Settings.embed_model = MockEmbedding(embed_dim=8)
    config = make_config()
    write_records(RECORDS, config.processed_records_path)
    return config


def _indexed_texts(config) -> set[str]:
    return set(get_collection(config).get(include=["documents"])["documents"])


def _bm25_texts(config) -> set[str]:
    return {metadata_dict_to_node(entry).get_content() for entry in load_bm25(config, top_k=1).corpus}


def test_first_run_is_a_full_build(config) -> None:
    report = ingest(config)
    assert (report.mode, report.reason) == ("full", "no existing vector index")
    assert (report.added, report.nodes_embedded, report.nodes_total) == (3, 3, 3)


def test_unchanged_corpus_embeds_nothing(config) -> None:
    ingest(config)
    report = ingest(config)
    assert report.mode == "incremental"
    assert (report.unchanged, report.nodes_embedded, report.nodes_total) == (3, 0, 3)


def test_changed_record_replaces_its_vectors_and_bm25_text(config) -> None:
    ingest(config)
    edited = [*RECORDS[:2], _record("Low-voltage detection triggers below 4.65V.", "lines:5-5")]
    write_records(edited, config.processed_records_path)

    report = ingest(config)

    # Record IDs hash the text, so an edit is an added document plus a removed one.
    assert (report.added, report.removed, report.unchanged, report.nodes_embedded) == (1, 1, 2, 1)
    assert "Low-voltage detection triggers below 4.65V." in _indexed_texts(config)
    assert "Low-voltage detection triggers below 4.63V." not in _indexed_texts(config)
    assert _bm25_texts(config) == _indexed_texts(config)


def test_removed_record_is_deleted_from_vectors_and_bm25(config) -> None:
    ingest(config)
    write_records(RECORDS[:2], config.processed_records_path)

    report = ingest(config)

    assert (report.removed, report.nodes_embedded, report.nodes_total) == (1, 0, 2)
    assert get_collection(config).count() == 2
    assert load_manifest(config)["node_count"] == 2
    assert "Low-voltage detection triggers below 4.63V." not in _bm25_texts(config)


def test_vector_policy_change_forces_a_full_rebuild(config) -> None:
    ingest(config)
    report = ingest(replace(config, chunk_size=256))
    assert report.mode == "full"
    assert report.reason.startswith("vector policy changed")
    assert report.nodes_embedded == 3


def test_index_without_a_document_store_is_rebuilt_in_full(config) -> None:
    ingest(config)
    shutil.rmtree(pipeline_dir(config))
    report = ingest(config)
    assert report.mode == "full"
    assert "no ingestion document store" in report.reason
    assert get_collection(config).count() == 3


def test_requested_full_rebuild_keeps_node_ids_and_fingerprint_stable(config) -> None:
    ingest(config)
    first = load_manifest(config)["corpus_sha256"]
    report = ingest(config, full_rebuild=True)
    assert (report.mode, report.reason, report.nodes_embedded) == ("full", "requested", 3)
    assert load_manifest(config)["corpus_sha256"] == first
