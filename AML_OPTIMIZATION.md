# AML Optimization Notes

Snapshot date: 2026-08-20. Scores below are public academic textual leaderboard
results, not results produced by the v0.3 code in this repository.

| System | Rank | Overall | Return size | Write | Search |
| --- | ---: | ---: | --- | --- | --- |
| InvMem | 1 | 45.057 | Balanced | Fast | Fast |
| ReFind | 2 | 44.974 | Verbose | Fast | Fast |
| LightMem-StuctMem | 43 | 35.476 | Compact | Fast | Slow |

The largest capability gaps from LightMem-StuctMem to InvMem are A
(`38.730 -> 56.775`) and B (`34.864 -> 45.956`). StructMem is already ahead on
E (`55.643 vs 53.139`) and H (`44.516 vs 30.323`). This suggests the next run
should prioritize broad evidence recall and exact matching without discarding
the old result as a useful structured-memory ablation.

## What The Leading Systems Do

ReFind stores raw Add chunks, reconstructs adjacent conversation records,
searches BM25 at conversation and session granularity, and uses `gpt-4o-mini` as
an iterative keyword/date search planner. It expands selected evidence with two
neighboring chunks on both sides and does not generate the final answer.

InvMem stores raw messages and their embeddings, fuses dense and BM25 rankings,
includes multiple-choice options in the retrieval query, and expands high-ranked
messages with same-session neighbors. Its documented frozen v0.6 settings use a
lexical weight of `0.5`, RRF `k=60`, a one-message result window, and 20 seeds.

## Problems In The Previous Submission

- The extraction prompts explicitly require LoCoMo topic markers, numeric
  `source_id` values, two-person relational behavior, and LoCoMo examples.
- Unknown roles are rewritten to `user`, losing speaker identity.
- Missing source timestamps are replaced with ingestion time, which invents
  temporal evidence.
- Search ignores the AML `options` field.
- Relationship extraction creates many low-discrimination memories such as
  congratulations and acknowledgements.
- Cross-event summaries rewrite exact names, numbers, and wording, then compete
  with event memories in one score list.
- Search can return `top_k + ceil(top_k / 5)` items instead of respecting the
  requested cap.

## v0.3 Hypothesis

The v0.3 service removes generation from Add, preserves exact evidence, and uses
option-aware Dense + BM25/RRF retrieval with local and cross-session context.
The expected benefit is higher A/B recall, lower write latency, and fewer
failures on names, codes, dates, and candidate-answer wording. This remains a
hypothesis until a new official AML run is completed.

The declared `BAAI/bge-small-en-v1.5` model was downloaded at Hugging Face
revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` and loaded with networking
disabled. A full LoCoMo10 retrieval replay used 13,340 raw messages, 1,977
questions with resolvable evidence, and `top_k=100`:

| Retrieval | Evidence Recall@100 |
| --- | ---: |
| Dense | 0.8355 |
| Dense + BM25/RRF | 0.8399 |
| Dense + BM25/RRF + one-message window | 0.8867 |

The one-message window improved every LoCoMo question category over the
corresponding no-window hybrid retrieval. A lexical-weight sweep with the
window enabled produced `0.8735` (dense only), `0.8760` (`0.1`), `0.8863`
(`0.25`), `0.8867` (`0.5`), `0.8838` (`0.75`), and `0.8791` (`1.0`). The
declared `0.5` remains the default because it has the best overall recall and
matches InvMem's frozen v0.6 configuration; `0.25` is the next cross-dataset
ablation because it retained more inference-category evidence. These are
retrieval proxy results, not official AML answer scores.

## Lightweight Structure Selection

The structure ablation adds at most one non-generated, cross-session candidate
for each of the first 10 retrieval seeds. A semantic bridge uses the existing
message vectors; a rare-anchor bridge uses low-document-frequency tokens. Both
are one-hop expansions, and returned items remain raw messages.

On the same full LoCoMo replay, the structure variants produced:

| Variant | Evidence Recall@100 |
| --- | ---: |
| Raw hybrid + local window | 0.8867 |
| + semantic bridge | **0.8882** |
| + rare-anchor bridge | 0.8878 |
| + both bridges | 0.8881 |

To check cross-dataset behavior without running the entire 265 MB benchmark, a
fixed 10% stratified LongMemEval sample was selected with seed `20260820`. It
contains 50 of 500 questions with counts `8/13/6/3/7/13` across knowledge
update, multi-session, assistant, preference, user, and temporal reasoning.
The metric is answer-session recall rather than generated-answer accuracy:

| Return budget | Baseline recall / hit | Semantic recall / hit | Anchor recall / hit | Combined recall / hit |
| --- | ---: | ---: | ---: | ---: |
| 20 | 0.9267 / 0.9600 | **0.9417 / 0.9800** | 0.9250 / 0.9800 | 0.9250 / 0.9800 |
| 50 | 0.9550 / 0.9800 | **0.9583 / 0.9800** | 0.9483 / 0.9800 | **0.9583 / 0.9800** |
| 100 | 0.9900 / 1.0000 | 0.9900 / 1.0000 | 0.9900 / 1.0000 | 0.9900 / 1.0000 |

The semantic bridge is the only structural module enabled by default: it
improves LoCoMo and the smaller LongMemEval return budgets without regressing
`@100`. The anchor module is excluded because it regresses LongMemEval recall at
`@20` and `@50`. The semantic bridge threshold is `0.72`; it reuses stored
vectors, adds no embedding or LLM request, and can be disabled with
`MEMORY_SEMANTIC_BRIDGE_SEED_K=0`.

Recommended ablations use the same ingestion and question set:

1. Dense only: `MEMORY_LEXICAL_WEIGHT=0`, `MEMORY_RESULT_WINDOW=0`.
2. Dense + BM25/RRF: default lexical settings, window disabled.
3. Dense + BM25/RRF + local context: semantic seed count set to `0`.
4. Dense + BM25/RRF + local and semantic context: the frozen v0.3 defaults.
5. Option removal: strip `options` only at Search to measure selection-task gain.

Record evidence recall, answer accuracy, returned characters, Add/Search P95,
and embedding cost for every variant. Do not tune against private AML payloads.
