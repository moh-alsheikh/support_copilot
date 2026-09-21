"""Coordinate retrieval, grounded synthesis, and deterministic citations."""

from support_copilot.generation.citations import citations_from_nodes
from support_copilot.generation.synthesizer import synthesize
from support_copilot.models import QueryResult
from support_copilot.retrieval.retriever import retrieve


class QueryService:
    """Small service boundary shared by command-line and Streamlit interfaces."""

    def __init__(self, config):
        """Store immutable application configuration."""

        self.config = config

    def answer(self, question: str, top_k: int | None = None, *, mode: str = "hybrid") -> QueryResult:
        """Answer from retrieved evidence using the requested retrieval strategy."""

        evidence = retrieve(self.config, question, top_k=top_k, mode=mode)
        answer = synthesize(question, evidence)
        return QueryResult(
            answer=answer,
            citations=citations_from_nodes(evidence),
            evidence=evidence,
        )
