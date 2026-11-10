"""Reranking logic with an injected fake scorer, so tests never download a model."""

from llama_index.core.schema import NodeWithScore, TextNode

from support_copilot.ingestion.document_adapter import load_documents
from support_copilot.ingestion.processed_records import write_records
from support_copilot.retrieval.reranker import rerank


def _candidate(node_id: str, text: str, rrf_score: float) -> NodeWithScore:
    node = TextNode(
        id_=node_id,
        text=text,
        metadata={"title": "Led Warning Flash Codes", "content_hash": "f" * 64, "semantic_rank": 3, "bm25_rank": 1},
        excluded_embed_metadata_keys=["content_hash", "semantic_rank", "bm25_rank"],
    )
    return NodeWithScore(node=node, score=rrf_score)


def _fused():
    return [
        _candidate("intro", "LED warning codes overview", 0.030),
        _candidate("row-0-7", "Long flashes: 0 | Short flashes: 7 | Status: Kernel image not found", 0.020),
    ]


def test_scorer_order_replaces_fusion_order_and_keeps_rank_diagnostics() -> None:
    reranked = rerank(_fused(), "0 long and 7 short flashes", lambda q, passages: [0.1, 0.9])
    assert [item.node.node_id for item in reranked] == ["row-0-7", "intro"]
    assert reranked[0].score == 0.9
    assert reranked[0].node.metadata["rrf_score"] == 0.020
    assert (reranked[0].node.metadata["semantic_rank"], reranked[0].node.metadata["bm25_rank"]) == (3, 1)


def test_scorer_sees_title_and_text_but_not_hashes() -> None:
    seen = []
    rerank(_fused(), "question", lambda q, passages: seen.extend(passages) or [0.0] * len(passages))
    assert seen[1].startswith("title: Led Warning Flash Codes")
    assert "Kernel image not found" in seen[1]
    assert "f" * 64 not in seen[1]


def test_no_scorer_keeps_fusion_order() -> None:
    assert [item.node.node_id for item in rerank(_fused(), "question", None)] == ["intro", "row-0-7"]
    assert rerank([], "question", lambda q, p: []) == []


def test_documents_embed_only_title_metadata(make_config) -> None:
    config = make_config()
    write_records(
        [{"doc_id": "DOC_0001", "text": "Row text", "metadata": {"source": "a.md", "location": "lines:1-1", "title": "A Guide", "content_hash": "f" * 64, "encoding": "utf-8-sig"}}],
        config.processed_records_path,
    )
    from llama_index.core.schema import MetadataMode

    embed_view = load_documents(config)[0].get_content(metadata_mode=MetadataMode.EMBED)
    assert embed_view == "title: A Guide\n\nRow text"
