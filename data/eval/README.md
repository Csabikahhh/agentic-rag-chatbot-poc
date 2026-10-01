# Evaluation data

This directory holds the question set of the functional evaluation and the committed reports of the evaluation and load-test runs (plan section 5.6 in [`docs/project-structure-plan.md`](../../docs/project-structure-plan.md)).

| Path | Content | Written in |
|---|---|---|
| `questions.jsonl` | 10–20 evaluation questions with reference answers, expected documents and expected intents | Phase 7 |
| `results/` | JSON reports of `agentic-rag eval` and `agentic-rag loadtest` | Phases 7 and 8 |

> **Status:** `questions.jsonl` does not exist yet. The 10–20 real questions are written in Phase 7, once the domain and the corpus are chosen (plan decision 8): reference answers and expected documents only make sense for a known corpus. The schema below is already enforced by the loader.

## `questions.jsonl` schema

UTF-8 [JSON Lines](https://jsonlines.org/): one JSON object per line, one line per question. `agentic_rag.evaluation.dataset.load_dataset` reads the file and validates every line as an `EvalItem`.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `id` | string | yes | Unique id such as `q01`: letters, digits, `.`, `_` and `-`, starting with a letter or a digit |
| `question` | string | yes | The question as a user would ask it |
| `reference_answer` | string | yes | An answer a domain expert accepts as correct; answer correctness is judged against it |
| `expected_documents` | list of strings | no, default `[]` | The documents that contain the answer, each written exactly as ingestion identifies it (`DocumentMetadata.source` in `agentic_rag.ingestion.loaders`, which `Source.source` repeats): its path relative to `DATA_DIR` (`data/raw` by default) with forward slashes, such as `guide/intro.md`, or its URL. Empty when the question needs no retrieval. Retrieval hit@k is computed against them |
| `expected_intent` | `"direct"`, `"single"`, `"complex"`, `"tool"` or `null` | no, default `null` | The route `analyze_request` should choose; `null` leaves routing unchecked for the question. Routing accuracy is computed against it |
| `tags` | list of strings | no, default `[]` | Free labels for grouping the results, e.g. `multi-part` or `out-of-scope` |
| `notes` | string or `null` | no, default `null` | Free text for reviewers, e.g. why the question is in the set |

An illustrative line, with domain-neutral placeholders and every field filled in:

```jsonl
{"id": "q01", "question": "<a question a user of the chosen domain would ask>", "reference_answer": "<the answer an expert accepts, in one or two sentences>", "expected_documents": ["<path, relative to data/raw, of the document that answers it>"], "expected_intent": "single", "tags": ["single-hop"], "notes": "<why the question is in the set>"}
```

The loader skips blank lines, strips leading and trailing whitespace from strings and accepts a UTF-8 byte order mark. It stops at the first invalid line and names the file and the line number (`data/eval/questions.jsonl:4: ...`) for any of these problems:

- text that is not UTF-8, invalid JSON, or a line that is not a JSON object;
- a key that occurs twice, or an unknown key (a typo such as `expected_document` fails instead of being ignored);
- a missing or empty required field, an unknown intent, or a value of the wrong type;
- the same entry twice in `expected_documents` or `tags`;
- an `id` already used on an earlier line.

To check the file after editing it:

```bash
uv run python -c "from pathlib import Path; from agentic_rag.evaluation.dataset import load_dataset; print(len(load_dataset(Path('data/eval/questions.jsonl'))), 'questions')"
```

## Writing the questions (Phase 7)

- Cover every route: `direct` (no retrieval needed), `single` (one lookup), `complex` (a multi-part question that the planner splits into sub-tasks) and `tool` (the non-retrieval tool).
- Take every answer from the corpus and keep the reference answers short and factual, so that a judge can compare them in substance.
- Write `expected_documents` exactly as ingestion identifies the documents: hit@k compares the identifiers exactly, so `setup.pdf` does not match a corpus file stored as `guides/setup.pdf`.
- Add at least one question the corpus cannot answer, to check that the system says so instead of inventing an answer.

## Results (`results/`)

`agentic-rag eval` and `agentic-rag loadtest` write one JSON report per run to this directory; `--output-dir` selects another one. Every report starts with `created_at` and a snapshot of the settings (`agentic_rag.reports.RunReport`). The report formats are defined in code:

- `eval` writes an `EvalReport` (`agentic_rag.evaluation.runner`): the target (the full graph, or one of the nodes an item can drive on its own, `NODE_TARGETS`: `analyze_request` for routing, `run_rag_subtask` for retrieval), the dataset path, one result per question and the aggregate scores (routing accuracy, retrieval hit@k, answer correctness, faithfulness), each with the number of questions it applies to. A result holds the answer, the chosen intent, the retrieved documents, the metric values, the latency and the error, and its verdicts must agree with the question: no routing verdict without an expected intent, no hit@k without expected documents.
- `loadtest` writes a `LoadTestReport` (`agentic_rag.loadtest.runner`): the request count, the concurrency, the error count and rate, the throughput, the end-to-end latency (count, mean, min, p50, p95, p99, max), the latency of every node, and the warm-up requests summarized separately.

hit@k is computed per retrieve sub-task. `retrieved_documents` holds one list per retrieve sub-task of the run: the documents of the sub-task's chunks (`SubtaskResult.sources`) in rank order. These are the chunks the RAG subgraph keeps after grading, so hit@k scores retrieval and grading together: a relevant chunk that the grading drops counts as a miss. A question scores a hit when at least one of its retrieve sub-tasks ranks a chunk of an expected document among its first `TOP_K`. The answer's de-duplicated citations (`AgentOutput.sources`) are never used, because their order is not a ranking.

The file names are fixed in Phases 7 and 8; the plan is to include the kind of run and a UTC timestamp, so that runs never overwrite each other. The reports behind the numbers in `docs/evaluation.md` and `docs/performance.md` are committed, so every documented number has a result file and a command that reproduces it (plan section 10).

## Running the evaluation and the load test

Run both on the host: `uv run agentic-rag eval` and `uv run agentic-rag loadtest` read the question set and write their reports to this directory directly. This is the recommended way.

The Compose stack mounts only `data/raw` into the `app` container. To run the commands in the container instead, also bind-mount `./data/eval` at `/app/data/eval`. The container runs as UID and GID 10001 by default, and a bind mount keeps the owner of the host directory, so the report can only be written when `results/` is writable by that user; otherwise the run ends with a `PermissionError`. Docker Desktop on Windows and macOS hides this, because it presents bind mounts as writable for everyone; a Linux engine enforces it. On Linux, either make `data/eval/results` writable for UID 10001, or build the image with your own IDs: `APP_UID=$(id -u) APP_GID=$(id -g) docker compose build app`. The named volumes `chroma-data` and `hf-cache` keep the owner they were created with, so after changing the IDs either change their owner in place (the command is in the comment on the `app` service in `compose.yaml`) or recreate them with `docker compose down -v`, which also deletes the index and the downloaded models.
