"""Expose semantic, BM25, and hybrid retrieval through one application interface."""

from llama_index.core.schema import NodeWithScore

from support_copilot.indexing.index_manager import load_bm25, load_index
from support_copilot.retrieval.reranker import cross_encoder_scorer, rerank

DEFAULT_CANDIDATE_K = 50


def resolve_candidate_k(top_k: int) -> int:
    """Return a broad pool for RRF while validating the final result count."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    return max(DEFAULT_CANDIDATE_K, top_k * 5)


def reciprocal_rank_fusion(result_lists, top_k: int, rank_constant: int = 60):
    """Fuse candidate positions and retain every branch rank for later inspection."""

    scores = {}
    nodes = {}
    branch_ranks: list[dict[str, int]] = []
    for results in result_lists:
        ranks = {}
        for rank, item in enumerate(results, start=1):
            node_id = item.node.node_id
            nodes[node_id] = item.node
            ranks[node_id] = rank
            scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (rank_constant + rank)
        branch_ranks.append(ranks)
    node_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
    output = []
    for node_id in node_ids:
        # Clone the node before adding query-time diagnostics. Persisted source metadata remains
        # unchanged, while the UI and reports can explain why this result ranked where it did.
        node = nodes[node_id].model_copy(deep=True)
        node.metadata = dict(node.metadata)
        node.metadata["semantic_rank"] = branch_ranks[0].get(node_id) if branch_ranks else None
        node.metadata["bm25_rank"] = branch_ranks[1].get(node_id) if len(branch_ranks) > 1 else None
        output.append(NodeWithScore(node=node, score=scores[node_id]))
    return output


def retrieve(config, question: str, top_k: int | None = None, *, mode: str = "hybrid"):
    """Retrieve evidence with a selected strategy and a larger hybrid candidate pool."""

    final_k = top_k if top_k is not None else config.top_k
    candidate_k = resolve_candidate_k(final_k)

    vector_results = load_index(config).as_retriever(similarity_top_k=candidate_k).retrieve(question)
    if mode == "semantic":
        return vector_results[:final_k]

    bm25_results = load_bm25(config, top_k=candidate_k).retrieve(question)
    if mode == "bm25":
        return bm25_results[:final_k]
    if mode != "hybrid":
        raise ValueError("mode must be semantic, bm25, or hybrid")
    fused = reciprocal_rank_fusion([vector_results, bm25_results], candidate_k)
    # An empty RERANKER_MODEL keeps fusion order, e.g. on a machine that can't download the model.
    scorer = (
        cross_encoder_scorer(config.reranker_model, config.storage_dir / "models")
        if config.reranker_model
        else None
    )
    return rerank(fused, question, scorer)[:final_k]
