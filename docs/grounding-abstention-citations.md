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
Chunk ID: manual-p7-c1
Content:
The inspection interval is 100 operating hours.
```

The application builds Citation 2 from the second retrieved node's metadata. It does not ask the
model to remember a filename. This makes the mapping deterministic and testable.

Do not deduplicate and renumber citation rows after creating evidence labels. If Sources 1 and 2
share a location and Source 3 is different, renumbering Source 3 as citation 2 makes every later
answer marker ambiguous.

## What Automatic Checks Can Prove

The program can prove that a cited source number exists, the citation maps to retrieved evidence,
the source/location exists in metadata, and an unsupported question followed abstention policy.
It cannot prove that evidence truly supports a nuanced engineering claim. Faithfulness and
citation accuracy still require human review or a separately validated judge as a secondary signal.

## Code Reading Path

Read this project's grounding implementation in this order:

1. `generation/prompts.py`
2. `generation/synthesizer.py`
3. `generation/citations.py`
4. `services/query_service.py`
5. `evaluation/evaluators.py`
6. `tests/test_citations.py`

Trace one `[Source N]` through every file. If the number changes meaning, the implementation is
wrong even when the answer sounds fluent.
