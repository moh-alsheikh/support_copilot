# Grounding, Abstention, And Citation Engineering

## Grounding Is A System Contract

A short instruction such as “use the evidence” is not enough for a technical assistant. Grounding
crosses several layers:

```text
retrieved NodeWithScore objects
  -> numbered evidence blocks
  -> QA and refine prompt rules
  -> model response
  -> abstention normalization
  -> code-attached [Source N] markers (sentence-level support check)
  -> metadata-derived citation rows
  -> answer/citation evaluation
```

Failure in any layer can create an answer that looks grounded but is not verifiable.

## One Exact Abstention Message

The system stores one `ABSTENTION_MESSAGE` constant. This matters because prompts, the pipeline,
tests, Streamlit, and evaluation all refer to one observable behavior. The pipeline also normalizes
a response that starts with the exact message and then appends a claim. Without normalization, a
model can say “I do not know” and then contradict itself.

Empty retrieval is handled before generation. Calling a model with no evidence wastes time and
creates an unnecessary opportunity to hallucinate.

## QA And Refine Prompts

LlamaIndex `compact` synthesis may pack evidence into one prompt or refine an answer when the input
does not fit. Supplying only `text_qa_template` leaves the refine stage free to use framework
defaults. This project supplies both templates explicitly so long contexts do not silently weaken
the rules.

Both templates require evidence-only answers, no outside knowledge, `[Source N]` citations, only
labels present in context, exact abstention, and concise technical output.

## Evidence Labels And Metadata

Before synthesis, each node becomes a visible block:

```text
[Source 2]
File: reports/manual.pdf
Location: page:7
Content:
The inspection interval is 100 operating hours.
```

Nothing else is in the block. Every metadata key is excluded from the LLM view of the prompt node;
otherwise LlamaIndex prints all of it (hashes, IDs, ranking scores) above each block, where it
reads as the tail of the previous source (`known-findings.md`, F5).

The application builds Citation 2 from the second retrieved node's metadata. It does not ask the
model to remember a filename. This makes the mapping deterministic and testable.

## Code-Attached Citations

Which *sentence* is backed by which source is also decided in code, not left to the model
(`generation/attribution.py`, called by `QueryService.answer()`). After generation, every sentence
is checked against the text of each evidence block (its title plus the content shown to the model,
since an answer may name the document it read):

- every value it states (`600mA`, `0°C`, `8000 RPM`) must appear in the source, and
- with three or more content words, at least half of them must appear there.

| Sentence | What the user sees |
|---|---|
| Supported by the source the model cited | Kept, with that marker |
| Supported by a different source, or the model cited nothing | Cited to the best-supporting source |
| A combined claim supported only by the sources the model named together | Cited to all of them |
| Supported by no retrieved source | Labelled `[Unverified]`; the UI and CLI show a warning |
| A connective with no values and few words ("Yes.", "Here are the steps:") | Left as is |

Markers are always placed by code at the end of the sentence, and phrases that only carried the
model's markers ("According to [Source 2], …", "[Source 1] states that …", "Evidence: [Source 1]")
are removed with them; a marker used as a subject ("[Source 3] recommends …") becomes "The source
recommends …". The raw model answer is kept as `QueryResult.model_answer` and shown in a
collapsed panel.

The check is lexical: deterministic and explainable, but blind to meaning. A heavy paraphrase that
shares few words with its source is labelled unverified, and a sentence that reuses a source's words
to say something it doesn't mean can still be cited. An answer whose sentences are all unverified is
still shown (labelled), not converted to an abstention.

Do not deduplicate and renumber citation rows after creating evidence labels. If Sources 1 and 2
share a location and Source 3 is different, renumbering Source 3 as citation 2 makes every later
answer marker ambiguous.

## What Automatic Checks Can Prove

The program can prove that a cited source number exists, the citation maps to retrieved evidence,
the source/location exists in metadata, every cited sentence passes the lexical support check, and
an unsupported question followed abstention policy.
It cannot prove that evidence truly supports a nuanced engineering claim. Faithfulness and
citation accuracy still require human review or a separately validated judge as a secondary signal.

## Code Reading Path

Read this project's grounding implementation in this order:

1. `generation/prompts.py`
2. `generation/synthesizer.py`
3. `generation/attribution.py`
4. `generation/citations.py`
5. `services/query_service.py`
6. `evaluation/evaluators.py`
7. `tests/test_citations.py`, `tests/test_attribution.py`

Trace one `[Source N]` through every file. If the number changes meaning, the implementation is
wrong even when the answer sounds fluent.
