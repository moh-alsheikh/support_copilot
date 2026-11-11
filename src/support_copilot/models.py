"""Application-owned result types keep the UI independent of framework internals."""

from dataclasses import dataclass, field


@dataclass
class Citation:
    """One unique source/location pair derived from retrieved evidence."""

    number: int
    source: str
    location: str
    chunk_id: str


@dataclass
class QueryResult:
    """Everything the CLI, evaluator, or Streamlit UI needs from one answer.

    `answer` carries `[Source N]` markers attached by code (`generation/attribution.py`);
    `model_answer` is the model's raw text, kept for diagnostics.
    """

    answer: str
    citations: list[Citation]
    evidence: list
    model_answer: str = ""
    claims: list = field(default_factory=list)

    @property
    def unverified_claims(self) -> list[str]:
        """Answer sentences no retrieved source supports."""

        return [claim.text for claim in self.claims if claim.status == "unverified"]
