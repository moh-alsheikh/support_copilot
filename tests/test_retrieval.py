"""Test persistent indexing and evaluation without Ollama or internet access."""

from llama_index.core import Settings
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import NodeWithScore, TextNode

from support_copilot.evaluation.evaluators import evaluate_retrieval, summarize
from support_copilot.indexing.index_manager import ingest, load_manifest
from support_copilot.indexing.vector_store import get_collection
from support_copilot.ingestion.processed_records import write_records
from support_copilot.retrieval.retriever import reciprocal_rank_fusion, resolve_candidate_k


def _record(text: str, source: str, location: str) -> dict:
    return {
        "doc_id": source,
        "text": text,
        "metadata": {
            "source": source,
            "location": location,
            "page": None,
            "document_type": "text",
            "extraction_method": "test",
        },
    }


def test_repeated_full_build_is_idempotent(make_config) -> None:
    config = make_config()
    write_records(
        [_record("The inspection interval is 100 operating hours.", "manual.txt", "lines:1-1")],
        config.processed_records_path,
    )
    Settings.embed_model = MockEmbedding(embed_dim=8)
    ingest(config)
    first_count = get_collection(config).count()
    ingest(config)
    assert first_count > 0
    assert get_collection(config).count() == first_count
    assert load_manifest(config)["node_count"] == first_count


def test_rrf_rewards_agreement_by_rank() -> None:
    first = TextNode(id_="first", text="first")
    second = TextNode(id_="second", text="second")
    semantic = [NodeWithScore(node=first, score=0.01), NodeWithScore(node=second, score=0.99)]
    bm25 = [NodeWithScore(node=first, score=50.0)]
    fused = reciprocal_rank_fusion([semantic, bm25], 2)
    assert fused[0].node.node_id == "first"
    assert fused[0].node.metadata["semantic_rank"] == 1
    assert fused[0].node.metadata["bm25_rank"] == 1


def test_evaluation_uses_universal_locations() -> None:
    result = NodeWithScore(
        node=TextNode(text="evidence", metadata={"source": "manual.pdf", "location": "page:7"}),
        score=1.0,
    )
    case = {
        "expected_sources": ["manual.pdf"],
        "expected_locations": ["page:7"],
        "question_type": "supported",
    }
    row = evaluate_retrieval([result], case, top_k=1)
    assert row["hit_at_k"] == 1.0
    assert summarize([row])["mrr"] == 1.0


def test_evaluation_does_not_match_the_right_source_at_the_wrong_location() -> None:
    result = NodeWithScore(
        node=TextNode(text="wrong page", metadata={"source": "manual.pdf", "location": "page:6"}),
        score=1.0,
    )
    case = {
        "expected_sources": ["manual.pdf"],
        "expected_locations": ["page:7"],
        "question_type": "supported",
    }
    assert evaluate_retrieval([result], case, top_k=1)["hit_at_k"] == 0.0


def test_candidate_pool_preserves_prerequisite_policy() -> None:
    assert resolve_candidate_k(4) == 50
    assert resolve_candidate_k(12) == 60
