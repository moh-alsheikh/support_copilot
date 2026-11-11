"""Attach `[Source N]` markers in code, from what the evidence text actually supports.

The model's own markers are treated as hints, never as the citation. Every sentence of the
generated answer is checked against the text of each evidence block:

- every value it states (a number with an optional unit) must appear in the source, and
- when it has at least three content words, at least half of them must appear there.

A sentence that passes is cited with the source(s) that support it: the model's own markers when
code confirms them, otherwise the best-supporting source. A sentence nothing supports is labelled
`[Unverified]` instead of being given a guessed citation. Short connective sentences with no
values ("Yes.", "Here are the details:") are left as they are.

The check is lexical, so it is deterministic and explainable but cannot judge meaning: a paraphrase
that shares few words with its source is labelled unverified, and a sentence that reuses a
source's words to say something else can still pass. `evaluation/evaluators.py` uses the same
helpers, so the app and the evaluation agree on what "supported" means.
"""

import re
from dataclasses import dataclass, field

from support_copilot.generation.prompts import ABSTENTION_MESSAGE

UNVERIFIED_LABEL = "[Unverified]"
# A sentence whose content words mostly don't occur in the source is not supported by it.
SUPPORT_THRESHOLD = 0.5
MIN_CLAIM_WORDS = 3

# A number with an optional known unit ("5.1V", "600mA", "8000 RPM", "16GB", "70°C"). Only real
# units count, so "Pi 5 is" yields "5", not "5is". Matched after normalising case and spacing.
_UNITS = r"%|°c|°f|ma|mah|a|v|kv|mv|w|kw|mw|hz|khz|mhz|ghz|b|kb|mb|gb|tb|kbps|mbps|gbps|rpm|cfm|mm|cm|m|km|in|s|ms|h|min|c|f"
_VALUE = re.compile(rf"(?<![\w.])(\d+(?:[.,]\d+)?)\s?({_UNITS})?(?![a-z])", re.IGNORECASE)
# "page 1", "lines 161-161", "row 4": references to a location, not claimed values.
_LOCATION_REFERENCE = re.compile(r"\b(?:pages?|lines?|rows?|sections?)\s+\d+(?:\s*[-–]\s*\d+)?", re.IGNORECASE)
_WORD = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*")
_STOPWORDS = frozenset(
    """a an and are as at be been by can could do does for from had has have how if in into is it
    its may might of on or should so than that the their them then there these they this to was
    were what when where which while who will with would you your also not no yes source sources
    according states stated based provided evidence unverified""".split()
)
MARKER = re.compile(r"\[/?Source\s+\d+\]", re.IGNORECASE)
_LIST_PREFIX = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+")
# Naming the file ("According to the gpio-pinout-diagram.png") attributes a claim; it isn't one.
_FILENAME = re.compile(r"\b[\w-]+\.(?:pdf|png|jpe?g|md|json|csv|txt|docx?)\b", re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<!\bMr\.)(?<!\bMrs\.)(?<!\bMs\.)(?<!\bDr\.)(?<!\bvs\.)(?<!\be\.g\.)(?<!\bi\.e\.)(?<=[.!?])\s+")
# Phrases that only exist to carry the model's markers; removed along with them, since code places
# the citation. "According to [Source 1], page 2, X" -> "X"; "[Source 1] states that X" -> "X".
# One or more markers, with the "," / "and" between them ("[Source 1] and [Source 3]").
_MARKER_RUN = r"\[/?Source\s+\d+\](?:\s*(?:,|and|&)?\s*\[/?Source\s+\d+\])*"
_MARKER_PHRASES = (
    re.compile(
        rf"\b(?:according to|as (?:stated|shown|mentioned|noted) in|per|see)\s+{_MARKER_RUN}"
        r"(?:\s*,?\s*(?:pages?|lines?)\s+\d+(?:\s*[-–]\s*\d+)?)?\s*,?\s*",
        re.IGNORECASE,
    ),
    re.compile(
        rf"{_MARKER_RUN}\s*(?:also\s+)?(?:states?|says|confirms?|shows?|mentions?|notes?|indicates?|explains?)\s+(?:that\s+)?",
        re.IGNORECASE,
    ),
)
# A marker used as the subject of a sentence: "[Source 3] recommends X" -> "The source recommends X".
_MARKER_SUBJECT = re.compile(rf"^\s*{_MARKER_RUN}\s*(?=(?:also\s+)?(?!according\b)[a-z]+s\b)", re.IGNORECASE)
# A leftover heading such as "Evidence: [Source 1]" carries nothing once markers are removed.
_EMPTY_HEADINGS = frozenset({"evidence", "source", "sources", "reference", "references", "citation", "citations"})


def citation_markers(text: str) -> set[int]:
    """Return explicit `[Source N]` markers from generated prose."""

    return {int(value) for value in re.findall(r"\[Source\s+(\d+)\]", text, re.IGNORECASE)}


def values(text: str) -> set[str]:
    """Stated values, normalised: "5.1 V" -> "5.1v"; location references are ignored."""

    text = _LOCATION_REFERENCE.sub(" ", text)
    return {(number.replace(",", ".") + (unit or "")).lower() for number, unit in _VALUE.findall(text)}


def content_words(text: str) -> set[str]:
    """Lower-cased words of two or more characters that aren't stopwords."""

    text = _FILENAME.sub(" ", text.lower())
    return {word for word in _WORD.findall(text) if word not in _STOPWORDS and len(word) > 1}


def sentences(text: str) -> list[str]:
    """Split prose into sentences, one line at a time, without list numbering ("1. ")."""

    found = []
    for line in text.splitlines():
        line = _LIST_PREFIX.sub("", line)
        found.extend(part.strip() for part in _SENTENCE_END.split(line) if part.strip())
    return found


def strip_markers(sentence: str) -> str:
    """Remove `[Source N]` markers and the phrases that only carried them."""

    text = sentence
    for phrase in _MARKER_PHRASES:
        text = phrase.sub("", text)
    text = _MARKER_SUBJECT.sub("The source ", text)
    text = " ".join(re.sub(_MARKER_RUN, " ", text, flags=re.IGNORECASE).split())
    text = re.sub(r"\s+([,.;:!?])", r"\1", text).lstrip(",;: ")
    return text[:1].upper() + text[1:]


def support_text(node) -> str:
    """What a sentence is checked against: the source's title plus the content shown to the model.

    The prompt also shows the file name, so an answer may name the document ("the GPIO pinout
    diagram"); the title makes those words count as supported.
    """

    title = node.metadata.get("title") or ""
    return f"{title}\n{node.get_content()}" if title else node.get_content()


def unsupported_values(claim: str, source_text: str) -> list[str]:
    """Values the claim states that the source text doesn't contain.

    A bare number ("Pi 5") counts as present if the source has it with any unit ("5A").
    """

    source_values = values(source_text)
    source_numbers = {re.match(r"[\d.]+", value).group(0) for value in source_values}
    return sorted(
        value
        for value in values(claim)
        if value not in source_values and not (value.replace(".", "").isdigit() and value in source_numbers)
    )


def word_support(claim: str, source_text: str) -> float | None:
    """Share of the claim's content words found in the source; None if the claim is too short to judge."""

    words = content_words(claim)
    if len(words) < MIN_CLAIM_WORDS:
        return None
    return len(words & content_words(source_text)) / len(words)


def is_claim(sentence: str) -> bool:
    """A sentence that states a value or has enough content words to be checked."""

    return bool(values(sentence)) or len(content_words(sentence)) >= MIN_CLAIM_WORDS


def supported_by(claim: str, source_text: str) -> bool:
    """True when every stated value appears in the source and enough of its words do."""

    if unsupported_values(claim, source_text):
        return False
    overlap = word_support(claim, source_text)
    return overlap is None or overlap >= SUPPORT_THRESHOLD


@dataclass
class Claim:
    """One answer sentence and the sources code verified for it."""

    text: str
    status: str  # "cited" | "unverified" | "not_a_claim"
    sources: list[int] = field(default_factory=list)
    model_sources: list[int] = field(default_factory=list)


def attribute_sentence(sentence: str, evidence_texts: list[str]) -> Claim:
    """Decide which evidence blocks support one sentence."""

    model_sources = sorted(citation_markers(sentence))
    claim = strip_markers(sentence)
    if claim == ABSTENTION_MESSAGE:
        # Kept verbatim so abstention checks still see the exact message, even mid-answer.
        return Claim(claim, "not_a_claim", [], model_sources)
    if claim.rstrip(":").strip().lower() in _EMPTY_HEADINGS:
        claim = ""
    if not is_claim(claim):
        return Claim(claim, "not_a_claim", [], model_sources)

    numbers = range(1, len(evidence_texts) + 1)
    supporting = [n for n in numbers if supported_by(claim, evidence_texts[n - 1])]
    confirmed = [n for n in model_sources if n in supporting]
    if confirmed:
        return Claim(claim, "cited", confirmed, model_sources)
    if supporting:
        best = max(supporting, key=lambda n: (word_support(claim, evidence_texts[n - 1]) or 0.0, -n))
        return Claim(claim, "cited", [best], model_sources)
    # A sentence combining facts from several sources the model named together.
    valid_model_sources = [n for n in model_sources if 1 <= n <= len(evidence_texts)]
    if len(valid_model_sources) > 1:
        combined = " ".join(evidence_texts[n - 1] for n in valid_model_sources)
        if supported_by(claim, combined):
            return Claim(claim, "cited", valid_model_sources, model_sources)
    return Claim(claim, "unverified", [], model_sources)


def _render(claim: Claim) -> str:
    if claim.status == "not_a_claim":
        return claim.text
    label = "".join(f"[Source {n}]" for n in claim.sources) if claim.sources else UNVERIFIED_LABEL
    text = claim.text
    if text.endswith(":**") or text[-1:] not in ".!?:;":
        return f"{text} {label}"  # list items and bold headings keep their own ending
    return f"{text[:-1].rstrip()} {label}{text[-1]}"


def attribute(answer: str, evidence_texts: list[str]) -> tuple[str, list[Claim]]:
    """Rewrite the answer with code-verified citations, keeping its line structure.

    `evidence_texts[n - 1]` is the content shown to the model as `[Source n]`.
    """

    lines, claims = [], []
    for line in answer.splitlines():
        prefix = _LIST_PREFIX.match(line)
        line_claims = [attribute_sentence(sentence, evidence_texts) for sentence in sentences(line)]
        claims.extend(line_claims)
        rendered = " ".join(text for text in (_render(claim) for claim in line_claims) if text)
        if line.strip() and not rendered:
            continue  # a line that held only markers, e.g. "[Source 3]"
        lines.append((prefix.group(0) if prefix else "") + rendered)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip(), claims
