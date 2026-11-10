import json

import pytest
from llama_index.core import Settings
from llama_index.core.embeddings import MockEmbedding

from support_copilot.indexing.index_manager import BM25_TOKEN_PATTERN, ingest, load_bm25, manifest_path
from support_copilot.ingestion.processed_records import write_records


def _record(text: str, source: str, location: str) -> dict:
    # BM25 also indexes metadata, so sources must not share the query words being tested.
    return {
        "doc_id": source,
        "text": text,
        "metadata": {"source": source, "location": location, "document_type": "markdown", "extraction_method": "test"},
    }


@pytest.fixture
def built(make_config):
    config = make_config()
    write_records(
        [
            _record(
                "LED warning codes — Long flashes: 1 | Short flashes: 3 | Status: Generic failure to boot",
                "decoy.md",
                "lines:1-3",
            ),
            _record("Long flashes: 0 | Short flashes: 7 | Status: Kernel image not found", "right.md", "lines:5-7"),
        ],
        config.processed_records_path,
    )
    Settings.embed_model = MockEmbedding(embed_dim=8)
    ingest(config)
    return config


def test_reloaded_bm25_keeps_the_one_character_token_pattern(built) -> None:
    assert load_bm25(built, top_k=2).token_pattern == BM25_TOKEN_PATTERN


def test_single_digit_tokens_outweigh_a_shared_word(built) -> None:
    # Only the digits favour right.md; "LED" favours the decoy, so a tokenizer that drops
    # one-character tokens picks the wrong row.
    top = load_bm25(built, top_k=1).retrieve("What does an LED with 0 long flashes and 7 short flashes mean?")[0]
    assert top.node.metadata["source"] == "right.md"


def test_index_built_under_an_older_bm25_policy_is_rejected(built) -> None:
    path = manifest_path(built)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["bm25_policy_version"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="BM25 policy changed"):
        load_bm25(built, top_k=1)
