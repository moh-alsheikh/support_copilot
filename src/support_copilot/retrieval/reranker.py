"""Transparent lexical reranking baseline applied after hybrid fusion."""


def rerank(results, query: str):
    """Reward exact query-term coverage with a small, inspectable bonus."""

    terms = {term.lower().strip(".,?") for term in query.split() if len(term) > 2}
    for item in results:
        hits = sum(1 for term in terms if term in item.node.get_content().lower())
        item.score = float(item.score or 0) + hits * 0.001
    return sorted(results, key=lambda item: item.score or 0, reverse=True)
