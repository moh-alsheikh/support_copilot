"""Application-owned result types keep the UI independent of framework internals."""

from dataclasses import dataclass


@dataclass
class Citation:
    """One unique source/location pair derived from retrieved evidence."""

    number: int
    source: str
    location: str
    chunk_id: str


@dataclass
class QueryResult:
    """Everything the CLI, evaluator, or Streamlit UI needs from one answer."""

    answer: str
    citations: list[Citation]
    evidence: list
