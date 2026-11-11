# Known Findings

Real diagnosed bugs found while building this system, kept visible on purpose — a fix whose
origin story is lost is a fix someone deletes by accident later. Format per finding: **Symptom →
Diagnosis → Fix → Lesson.** Add new findings the same way: with evidence, before/after, and no
shame.

## F1. The candidate pool bug that defeated RRF

- **Symptom:** hybrid retrieval was never better than its best single branch.
- **Diagnosis:** the pre-fusion pool was capped near `top_k`. A chunk ranked 11th by both
  branches can never be fused into a top-10 — fusion had nothing to fuse.
- **Fix:** `candidate_k = max(50, top_k * 5)` in `resolve_candidate_k()`. It is not an arbitrary
  constant; it is a scar.
- **Lesson:** when you see a pool-size default in any framework's fusion retriever, check what it
  actually is. `QueryFusionRetriever` fuses whatever each branch's `similarity_top_k` returns —
  sizing that pool is still your decision, not the framework's.

## F2. Models answering after saying "I don't know"

- **Symptom:** `llama3.2:3b` emitted the exact abstention sentence, then appended an invented
  answer anyway.
- **Diagnosis:** prompting alone cannot guarantee output shape on small local models.
- **Fix:** the pipeline normalizes any response that *starts with* the abstention message down to
  the clean message (`generation/synthesizer.py` / the pipeline's normalization step).
- **Lesson:** an output contract needs an enforcement point in code, not only in the prompt.

## F3. Ground truth was wrong more often than retrieval

- **Symptom:** evaluation flagged "retrieval failures" for questions the system had actually
  answered from a correct page.
- **Diagnosis:** the source manual states several facts in two places (e.g. engine model on pages
  3 *and* 41). Ground truth credited only one location, so a correct retrieval scored zero.
- **Fix:** pair-aware ground truth listing every valid `(source, location)` pair per question —
  see `evaluation/dataset.py`'s validator and the position-aligned `expected_sources`/
  `expected_locations` arrays in the evaluation dataset.
- **Lesson:** before debugging the system, debug the answer key. Most "retrieval failures" turn
  out to be ground-truth bugs.

## F4. PDF text-layer artifacts break exact-string checks

- **Symptom:** the engine identifier "IO-360-L2A" never matched, though the right page was
  retrieved and the answer looked correct on screen.
- **Diagnosis:** the source PDF's text layer extracts it as "I0-360-L2A" — digit zero, not letter
  O. A font/extraction artifact, invisible to the eye.
- **Where it matters:** any exact-string answer check downstream (automated grading, evaluation
  scripts). Compare identifiers with artifact-tolerant logic or verify manually — don't trust
  visual inspection of an extracted string.

## F5. Node metadata leaked into the prompt and blurred the source labels

- **Symptom:** on the Raspberry Pi knowledge base, `llama3.2:3b` gave correct answers with no
  `[Source N]` marker at all (Q007, Q012, Q019 in run `answers-after-retrieval-upgrades`),
  although the right evidence was retrieved and labelled.
- **Diagnosis:** recording the exact prompt showed that `number_evidence()` copied each node's
  full metadata onto the new prompt node without excluding it from the LLM view, so LlamaIndex
  printed `source`, `doc_id`, `content_hash`, `semantic_rank`, `bm25_rank`, `rrf_score`, … above
  every `[Source N]` block. Each dump sat directly after the previous source's content, so it read
  as part of the wrong source, and about a third of the 7.4k-character prompt was noise. The
  metadata exclusions set at ingestion didn't help, because the prompt node is a fresh copy.
- **Fix:** `number_evidence()` excludes every metadata key from the LLM and embedding views and
  drops the chunk-ID hash line; the block is now label, file, location, content. Covered by
  `test_prompt_evidence_shows_no_metadata_dump`. Citations are also no longer left to the model:
  `generation/attribution.py` attaches them in code (see `grounding-abstention-citations.md`).
- **Lesson:** read the prompt the model actually receives, not the template. Framework defaults
  (here, metadata printing in `MetadataMode.LLM`) can rewrite it.
