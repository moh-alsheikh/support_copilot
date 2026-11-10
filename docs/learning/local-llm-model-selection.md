# Choosing a Local Generator Model for This Laptop

Which Ollama model should replace `llama3.2:3b` as the answer generator, given this project's
failure pattern and this machine's hardware. Researched 2026-10-01 against the Ollama library;
nothing below has been benchmarked yet except the current baseline.

## The problem a model swap can and can't fix

On the Raspberry Pi knowledge base (`evaluation/datasets/evaluation_questions.json`), the
`raspberry-pi-baseline` answer run with `llama3.2:3b` refused six supported questions. They split
into two different failures:

| Questions | Evidence retrieved? | Cause | Fixed by a better model? |
|---|---|---|---|
| Q001, Q007 | No | Retrieval: a chunking bug (Q001) and a rank-6 near miss (Q007) | **No.** Refusing was correct given the evidence. Fix retrieval instead |
| Q003, Q011 | Yes | The model misread or under-trusted clean text evidence | **Likely** |
| Q014, Q015 | Yes | Same, plus noisy OCR text (`0` read as `@`) from the two PNGs | **Partly.** OCR noise remains |

So a model swap targets Q003, Q011, Q014, and Q015 only. It must also not break what already
works: 3/3 correct refusals on the near-miss unsupported questions (Q016–Q018), and valid
citations on every answer.

## Hardware constraints

| Component | Value | What it means |
|---|---|---|
| CPU | Intel Core i7-4770HQ (2014, 4 cores, AVX2) | Ollama runs CPU-only on Intel Macs, so there's no GPU acceleration |
| RAM | 16 GB DDR3-1600 | Memory bandwidth, not core count, sets generation speed |
| OS | macOS 12 | — |
| Disk | ~21 GB free on the internal disk | Ollama stores models in `~/.ollama/models` |

Two consequences:

- **Reading the evidence is the slow part.** Each question sends ~2–3k tokens of retrieved
  evidence to the model. On CPU, the model has to process all of it before producing the first
  word, and bigger models process proportionally slower. The measured baseline, `llama3.2:3b`
  (2.0 GB), averages **53.5 s per answer, 119 s at worst**.
- **Memory budget ≈ 6 GB per model.** After macOS, apps, and `nomic-embed-text`, models up to
  ~6 GB fit comfortably. Larger ones start swapping.

Mixture-of-experts (MoE) models are the exception to "bigger is slower": they only run a small
subset of their parameters per token, so their speed tracks *active* parameters, not total size.

## Selection criteria

1. **Grounded reading:** follows "answer only from this evidence" instructions and answers when
   the evidence supports it, instead of refusing by default.
2. **Fits the memory and speed budget:** ≤ ~6 GB and, ideally, no slower than ~2× the baseline.
3. **Controllable thinking:** thinking models must run with thinking disabled (see below).
4. **Permissive license** where stated.

## Shortlist (requires `ollama pull`)

| Model | Download | Why it's a candidate | Expected speed vs. baseline ¹ | License |
|---|---|---|---|---|
| **`granite4:micro`** (3B) | 2.1 GB | IBM states it is built for RAG and question-answering over provided text. **Try first.** | ≈ same | Apache 2.0 |
| **`phi4-mini`** (3.8B) | 2.5 GB | Trained specifically for precise instruction adherence; widely cited as the strongest CPU model in its class. No RAG-specific claims | ≈ same to slightly slower | Not stated on the Ollama page |
| **`qwen3.5:4b`** | 3.4 GB | Strong instruction-following benchmark scores (IFEval, IFBench). **Thinking model**, so needs thinking off | ~1.5× slower | Not stated on the Ollama page |
| **`granite4:tiny-h`** (7B total, ~1B active) | 4.2 GB | MoE, RAG-oriented. Could be *faster* than the baseline despite its size | Unknown ² | Apache 2.0 |
| **`granite4.2:3b`** | 2.2 GB | Newer Granite, RAG-oriented. **Thinking is on by default** | ≈ same with thinking off | Apache 2.0 |

¹ Estimated from model size relative to the 2.0 GB baseline, not measured.
² It uses a newer hybrid (Mamba) architecture; how well Ollama's CPU backend handles it is
untested here.

### Already installed: test these at no download cost

| Model | Size | Role |
|---|---|---|
| `gemma3:4b` | 3.3 GB | Solid instruction-follower; ~1.5× slower |
| `qwen3.5:9b` | 6.6 GB | Quality ceiling within the memory budget; expect ~2–3 min per answer. **Thinking model** |

### Installed, but skip

| Model | Reason |
|---|---|
| `gpt-oss:20b` (13 GB) | Swaps with everything else on 16 GB |
| `gemma4:12b`, `gemma4:latest` (7.6–9.6 GB) | Roughly 4–5 min per answer on this CPU |
| `deepseek-r1:8b` | Always writes out its reasoning first; very slow on CPU |
| `llama2:7b`, `llama2:13b` | Outdated |
| `qwen2.5-coder:3b`, `qwen2.5-coder:7b` | Tuned for code, not grounded QA |

## Prerequisite for thinking models

`qwen3.5` and `granite4.2` produce a hidden reasoning trace before answering. The installed
`llama-index-llms-ollama` client returns that trace separately from the answer, so it won't break
the exact-abstention check. But generating it costs minutes per question on this CPU.

The client exposes a `thinking` parameter (`llama_index/llms/ollama/base.py`), but
`configure_llamaindex()` in `src/support_copilot/config.py` never sets it. Before evaluating a
thinking model, add a setting that passes `thinking=False` to `Ollama(...)`. This change is **not
made yet**. Non-thinking models (`granite4:micro`, `phi4-mini`, `gemma3:4b`) can be evaluated
without it.

## How to evaluate a candidate

Retrieval is identical for every model (same index, same `--mode` and `--top-k`), so any
difference comes from generation alone. The model is chosen by the `OLLAMA_LLM_MODEL` environment
variable, which overrides `.env` without editing it.

Run evaluations while the CPU is otherwise idle: a concurrent build or indexing job inflates
latency and makes models incomparable.

```bash
ollama pull granite4:micro

OLLAMA_LLM_MODEL=granite4:micro \
  python scripts/run_evaluation.py answers --mode hybrid --top-k 4 --run-name model-granite4-micro
```

Then compare `evaluation/reports/<run-name>/generated_answers.csv` against
`evaluation/reports/raspberry-pi-baseline/generated_answers.csv`, focusing on:

- `false_abstention` on Q003, Q011, Q014, Q015 (should drop)
- `false_answer` on Q016–Q018 (must stay `False`)
- `invalid_citation_markers` (must stay empty)
- `expected_evidence_cited` on answered questions
- `latency_seconds`

Then score the answers with the human rubric in `manual_answer_evaluation.csv`. The substring check
`answer_contains_expected` is a weak signal, especially for Q011 (a "No" answer).

### Decision rule

Prefer the model that removes the most false abstentions among Q003, Q011, Q014, and Q015,
**provided** it keeps zero false answers on Q016–Q018 and zero invalid citations. Break ties on
latency. A model that answers more but makes up a single answer to a near-miss question is worse
than the baseline, because refusing is this project's core safety property.

## Results

| Model | Supported answered (of 15) | False abstentions | Correct refusals (of 3) | Invalid citations | Avg / max latency |
|---|---|---|---|---|---|
| `llama3.2:3b` (baseline) | 9 | 6 (Q001, Q003, Q007, Q011, Q014, Q015) | 3 | 0 | 53.5 s / 118.8 s |
| `granite4:micro` | — | — | — | — | — |
| `phi4-mini` | — | — | — | — | — |
| `gemma3:4b` | — | — | — | — | — |
| `qwen3.5:4b` | — | — | — | — | — |

Baseline: run `raspberry-pi-baseline`, `--mode hybrid --top-k 4`.

## Sources

- [Ollama library — newest models](https://ollama.com/search?o=newest)
- [Ollama: granite4](https://ollama.com/library/granite4) and [granite4 tags](https://ollama.com/library/granite4/tags)
- [Ollama: granite4.2](https://ollama.com/library/granite4.2)
- [Ollama: phi4-mini](https://ollama.com/library/phi4-mini)
- [Ollama: qwen3.5](https://ollama.com/library/qwen3.5)
- [Best CPU-only local LLMs in 2026 (Popular AI)](https://www.popularai.org/p/best-cpu-only-local-llm-2026)
- [Best Ollama models in 2026 by task and hardware (LumaDock)](https://lumadock.com/tutorials/best-ollama-models)
- [Best Small Language Models 2026 (localaimaster)](https://localaimaster.com/blog/small-language-models-guide-2026)
