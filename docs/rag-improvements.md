# RAG reliability and retrieval update

This update implements the six review recommendations. The reports of 2026-10-03 in
`data/eval/results/` describe the previous pipeline and serve as its baseline; the reports
of 2026-10-04 measure the new defaults with the same models, judge and corpus (see
[Measured results](#measured-results)).

## Verification and tool answers

An unreadable verification response now produces `verdict="unavailable"`. The workflow
ends without claiming grounding and withholds the draft, asking the user to retry.
The ordinary `insufficient` verdict still uses the bounded re-plan loop.

A successful single built-in tool request needs only the router model call. The synthesis
node makes no model call, verification is skipped, and finalization displays the exact
tool output. The allowlist covers contrast, specificity and browser support. Failed tools,
unknown tools, multi-tool questions, and mixed retrieval/tool questions retain synthesis
and verification. Verbatim tool output retains the tool's language.

## Retrieval

`RETRIEVAL_CANDIDATES=20` sets the candidate pool; the effective pool is at least `TOP_K`.
`TOP_K=4` now caps the relevant chunks passed into the answer context after grading.

With `HYBRID_SEARCH=true`, a lazily built SQLite FTS5 index searches the same chunks as
Chroma using BM25, weighting titles and section names above body text. The lexical query
includes both the original question and the English rewrite, so rewriting cannot erase
API names. API punctuation such as `:has`, `$fetch`, underscores and hyphens is retained.
User tokens are quoted and SQL parameters are bound.

Reciprocal-rank fusion combines the semantic and lexical rankings and selects the candidate
pool before relevance grading. Fusion ranks are separate from cosine similarity: lexical-only
hits do not display a fabricated similarity score. Lexical matches can bypass the cosine
threshold but still go through the configured LLM grader. With grading disabled, this is
keyword/rank selection only; it is not a claim of semantic relevance or a confidence score.

The keyword index is built once per compiled graph, under a lock, with no extra model or
package dependency. Its first search pays the indexing cost. Restart the app after changing
the corpus to refresh the cached keyword snapshot. Keep the previous retrieval baseline with
`HYBRID_SEARCH=false RETRIEVAL_CANDIDATES=4 TOP_K=4`.

## Evaluation

The original 17 questions remain the smoke/regression set. The comparison questions now
also require evidence groups for both frameworks. Existing `hit@k` remains available for
comparison with old reports; `complete_evidence@k` requires every specified evidence group.
The new score is recorded per item in JSON and summarized in Markdown reports.

`data/eval/holdout.jsonl` is a separate 24-question acceptance set. It includes follow-ups
with explicit history, Hungarian questions, framework ambiguity, version boundaries,
unsupported APIs, and mixed tool/retrieval questions. Do not use it for prompt tuning;
if it becomes a development set, create another untouched holdout. Its draft reference
answers should be reviewed by a domain expert before treating it as a quality benchmark.

```powershell
# Run with the current local model and an independently chosen judge.
uv run agentic-rag eval --dataset data/eval/questions.jsonl --judge-model qwen2.5:7b-instruct
uv run agentic-rag eval --dataset data/eval/holdout.jsonl --judge-model qwen2.5:7b-instruct
uv run agentic-rag loadtest --requests 100 --concurrency 4
```

Compare retrieval coverage, correctness, faithfulness, refusal behavior, and p50/p95 latency.
Run baseline and hybrid configurations against the same corpus and model. A larger pool
can increase grading latency; do not assume it improves every query. The offline tests
verify behavior, not live answer quality.

## Measured results

Measured on 2026-10-04 on the machine of the baseline (RTX 5070 Laptop GPU, Ollama 0.35.0
with its defaults), with `qwen3.5:4b` (build `2a654d98e6fb`, thinking off) and the judge
`hf.co/bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M`; three evaluation runs with identical
scores, one load-test run per concurrency:

| Measure | Baseline (2026-10-03) | Update (2026-10-04) |
|---|---|---|
| Routing (17 questions) | 1.00 | 1.00 |
| Retrieval hit@4 | 0.90 | 1.00 |
| Complete evidence@4 | 1.00 (recomputed) | 1.00 |
| Correctness | 0.91 | 0.91 (0.97 without two judge errors) |
| Faithfulness | 0.94 | 1.00 |
| Retrieval alone: hit@4, median call | 0.90, 0.6 s | 1.00, 1.5 s |
| Load, one user: p50, p95 | 3.6 s, 14.9 s | 3.9 s, 12.9 s |
| Load, four users: throughput, p50, p95 | 12.2/min, 17.6 s, 35.4 s | 12.0/min, 20.4 s, 39.7 s |

With `HYBRID_SEARCH=false RETRIEVAL_CANDIDATES=4` the new code reproduces the baseline
retrieval (hit@4 0.90, 0.6 s), so the keyword ranking and the larger pool account for the
gain. The holdout (one run) scores routing 0.94, hit@4 0.81, complete evidence@4 1.00,
correctness 0.67 (0.79 without three judge errors) and faithfulness 0.73; its failures are
version boundaries, an ambiguous question answered for one framework, model errors and the
answer language. Details: [evaluation.md](evaluation.md#rag-reliability-update) and
[performance.md](performance.md#rag-reliability-update).

## Citations and migration

Numbered answer citations resolve to clickable HTTP(S) source URLs through Markdown
reference links; code samples remain untouched. The retrieved-context expanders also show
the indexed commit SHA when available. This is the corpus snapshot, not a claim that the
public documentation URL still serves that revision. Completed execution traces remain
collapsed by default; running traces are visible.

Ingestion now preserves each downloaded source's manifest commit in chunk metadata and
identity. Run `uv run agentic-rag ingest` once, then restart the frontend, to populate this
metadata in existing indexes. The standard `agentic-rag serve` startup does this ingestion
automatically. Adding revision metadata changes downloaded chunk IDs, so this first update
re-embeds those chunks; subsequent unchanged ingestions remain incremental. Legacy sources
without revision metadata continue to work and are not assigned an invented version.
