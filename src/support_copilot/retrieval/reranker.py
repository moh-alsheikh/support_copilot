"""Cross-encoder reranking applied after hybrid fusion.

Fusion ranks the ~50 candidates by where they appeared in the semantic and BM25 lists, so a chunk
that only one branch ranks highly (the exact LED table row for "0 long, 7 short flashes", which
semantic search places 43rd) loses to chunks both branches find vaguely relevant. A cross-encoder
reads the question and each candidate together and scores how well that candidate answers it.

The model runs locally through `fastembed` on ONNX Runtime (already required by Chroma), so no
PyTorch is needed. See docs/learning/bi-encoder-vs-cross-encoder.md.
"""

from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path

from llama_index.core.schema import MetadataMode

Scorer = Callable[[str, Sequence[str]], Sequence[float]]


@lru_cache(maxsize=None)
def _load_cross_encoder(model_name: str, cache_dir: str):
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=model_name, cache_dir=cache_dir)


def cross_encoder_scorer(model_name: str, cache_dir: Path) -> Scorer:
    """Return a scorer backed by a locally cached cross-encoder; the model loads on first use."""

    def score(query: str, passages: Sequence[str]) -> list[float]:
        return list(_load_cross_encoder(model_name, str(cache_dir)).rerank(query, list(passages)))

    return score


def rerank(results, query: str, scorer: Scorer | None):
    """Reorder fused candidates by cross-encoder relevance; keep fusion order when scorer is None.

    Each candidate is scored on its embed view (title plus text, without hashes or IDs). The fusion
    score is kept as `rrf_score` in node metadata beside `semantic_rank` and `bm25_rank`, so a
    ranking can still be explained after reranking. Fusion already cloned these nodes, so persisted
    metadata is untouched.
    """

    if scorer is None or not results:
        return list(results)
    passages = [item.node.get_content(metadata_mode=MetadataMode.EMBED) for item in results]
    scores = scorer(query, passages)
    for item, score in zip(results, scores):
        item.node.metadata["rrf_score"] = item.score
        item.score = float(score)
    return sorted(results, key=lambda item: item.score, reverse=True)
