# Load test and bottleneck analysis

> **Status:** Phase 8, 2026-10-03; re-measured on 2026-10-04 after the [RAG reliability update](rag-improvements.md), with the same machine, server settings and model builds. Every load-test number below comes from a report committed in [`data/eval/results/`](../data/eval/results/) (`loadtest-*.json`, each with a Markdown summary); the per-token micro-benchmark of the parallel slots was measured by hand and is marked as such.

## Contents

1. [Method](#method)
2. [Results](#results)
3. [Where the time goes](#where-the-time-goes)
4. [The bottleneck](#the-bottleneck)
5. [RAG reliability update](#rag-reliability-update)
6. [Optimization proposals](#optimization-proposals)
7. [Consequence for the model choice](#consequence-for-the-model-choice)
8. [Limitations](#limitations)
9. [Reproducing the runs](#reproducing-the-runs)

## Method

`agentic-rag loadtest` builds the compiled main graph once and sends the questions of the evaluation set ([`data/eval/questions.jsonl`](../data/eval/questions.jsonl)) to it in turn, so the load mixes single searches, comparisons with parallel searches, tool questions, a greeting and a question outside the topic in the proportions of the set. Three warm-up requests run first, one after the other, and are reported apart, because the first request loads the models. The measured requests come from a `ThreadPoolExecutor` with `concurrency` workers, every one calling `graph.invoke`, the same synchronous path as the UI; the latency of a request is its wall-clock time, and the per-node statistics come from the trace every request returns ([architecture.md](architecture.md#execution-model)).

Setup: Windows 11, NVIDIA RTX 5070 Laptop GPU (8 GB), Ollama 0.35.0 on the host, embeddings `intfloat/multilingual-e5-small` on the CPU in the load-test process, the full index, `TOP_K=4`, `MAX_RETRIES=2`, `OLLAMA_NUM_CTX=8192`. Models: Qwen2.5-7B-Instruct (the `hf.co/bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M` build of the former default) and Qwen3.5-4B (`qwen3.5:4b`) with `OLLAMA_REASONING=false`. The Ollama server ran with its defaults, one request at a time (`OLLAMA_NUM_PARALLEL=1`), unless a row says otherwise; the server settings are not part of the report files, so the table names them. Model builds, as `ollama list` shows them: `qwen3.5:4b` `2a654d98e6fb` and `hf.co/bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M` `eb180556ed65`. The rows U1 and U2 run the [RAG reliability update](rag-improvements.md) with its defaults (`RETRIEVAL_CANDIDATES=20`, `HYBRID_SEARCH=true`).

## Results

| Run | Model | Server | Concurrency | Requests | Throughput | Mean | p50 | p95 | p99 | Report |
|---|---|---|---|---|---|---|---|---|---|---|
| Fake baseline | Scripted fake | – | 4 | 100 | 4 520 / min | 0.05 s | 0.01 s | 0.06 s | 1.08 s | `loadtest-fake-c4-20261003T154729Z` |
| A1 | Qwen2.5-7B | 1 slot | 4 | 100 | 9.5 / min | 25.1 s | 20.2 s | 58.0 s | 65.2 s | `loadtest-ollama-c4-20261003T155933Z` |
| A2 | Qwen3.5-4B | 1 slot | 4 | 100 | 12.2 / min | 19.4 s | 17.6 s | 35.4 s | 50.1 s | `loadtest-ollama-c4-20261003T160836Z` |
| A3 | Qwen3.5-4B | 1 slot | 1 | 50 | 11.5 / min | 5.2 s | 3.6 s | 14.9 s | 15.9 s | `loadtest-ollama-c1-20261003T161348Z` |
| C1 | Qwen2.5-7B | 4 slots, flash attention | 4 | 100 | 20.0 / min | 11.9 s | 10.3 s | 25.0 s | 42.5 s | `loadtest-ollama-c4-20261003T165720Z` |
| C2 | Qwen3.5-4B, `GRADE_WITH_LLM=false` | 1 slot | 4 | 100 | 10.7 / min | 22.2 s | 18.2 s | 41.4 s | 64.3 s | `loadtest-ollama-c4-20261003T170820Z` |
| U1 | Qwen3.5-4B, RAG reliability update | 1 slot | 4 | 100 | 12.0 / min | 19.8 s | 20.4 s | 39.7 s | 42.7 s | `loadtest-ollama-c4-20261004T195038Z` |
| U2 | Qwen3.5-4B, RAG reliability update | 1 slot | 1 | 50 | 11.7 / min | 5.1 s | 3.9 s | 12.9 s | 15.0 s | `loadtest-ollama-c1-20261004T195534Z` |

No request failed in any run.

## Where the time goes

A3 sends one request at a time, so its node times are pure service times, without waiting in a queue. Per request (Qwen3.5-4B):

| Node | Kind | Runs per request | Mean per run | Share of the request time |
|---|---|---|---|---|
| `synthesize_answer` | LLM, writes the answer | 1.14 | 2.18 s | 47 % |
| `analyze_request` | LLM, routing | 1.00 | 0.94 s | 18 % |
| `run_rag_subtask` | RAG subgraph | 0.84 | 1.15 s | 18 % |
| ↳ `rewrite_query` | LLM | 0.84 | 0.87 s | 14 % |
| ↳ `grade_documents` | LLM | 0.84 | 0.26 s | 4 % |
| ↳ `retrieve` | Embedding and vector search | 0.84 | 0.02 s | 0.4 % |
| `verify_answer` | LLM | 1.14 | 0.54 s | 12 % |
| `plan_subtasks` | LLM | 0.38 | 1.03 s | 8 % |
| `call_tool` | Deterministic tools | 0.48 | 0.05 s | 0.4 % |
| `finalize_response` | Data | 1.00 | < 1 ms | 0 % |

(The subgraph nodes are part of `run_rag_subtask`'s time.) A request makes 5.3 LLM calls on average: the routing, the rewrite and the grade of each search, the answer and its verification, and, for 14 % of the requests, a re-plan with another round. Everything that is not an LLM call, the retrieval included, takes about 1 % of the time: the fake baseline, which runs the same graph with a scripted model, answers in 10 ms at the median. Its p99 of 1.1 s is the first `browser_support` call, which builds the index of the 2 344 browser-compat-data files once (concurrent first calls wait for it); later calls take milliseconds.

## The bottleneck

**The LLM server runs one request at a time.** Ollama's default is one slot per model (`OLLAMA_NUM_PARALLEL=1`), so with four users every LLM call waits for the calls ahead of it:

- From one to four concurrent requests (A3 to A2), the throughput grows by 6 % (11.5 to 12.2 per minute) while the median latency grows almost fivefold (3.6 s to 17.6 s).
- The same node takes several times longer under load: `rewrite_query` 0.87 s alone, 3.92 s with four requests in flight; `analyze_request` 0.94 s and 3.48 s.
- The GPU is busy all the time (88–99 % utilization during the runs), so the limit is the GPU's decoding rate for one sequence at a time, about 85 generated tokens per second for Qwen3.5-4B.

The embedding model, the vector search, the tools and LangGraph's orchestration are not bottlenecks.

## RAG reliability update

The [update](rag-improvements.md) changes the work per request in two directions: an exact single tool result skips the answer and its verification (two LLM calls), and every search grades up to 20 fused candidates instead of 4. U2 against A3, one request at a time, so service times without queueing:

| Node | Runs per request, A3 → U2 | Mean per run, A3 → U2 | Share of the request time, A3 → U2 |
|---|---|---|---|
| `synthesize_answer` | 1.14 → 1.10 | 2.18 s → 2.25 s | 47 % → 48 % |
| `run_rag_subtask` | 0.84 → 0.80 | 1.15 s → 1.71 s | 18 % → 27 % |
| ↳ `rewrite_query` | 0.84 → 0.80 | 0.87 s → 1.21 s | 14 % → 19 % |
| ↳ `grade_documents` | 0.84 → 0.80 | 0.26 s → 0.48 s | 4 % → 7.5 % |
| ↳ `retrieve` | 0.84 → 0.80 | 0.02 s → 0.02 s | 0.4 % → 0.3 % |
| `analyze_request` | 1.00 → 1.00 | 0.94 s → 0.88 s | 18 % → 17 % |
| `verify_answer` | 1.14 → 0.86 | 0.54 s → 0.46 s | 12 % → 7.7 % |
| `plan_subtasks` | 0.38 → 0.34 | 1.03 s → 1.01 s | 8 % → 6.7 % |

The verification runs 25 % less often: the 0.24 runs per request of `synthesize_answer` without a verification are the exact tool answers, which make no model call, so a request makes about 4.7 LLM calls instead of 5.3. The grading call takes 0.22 s longer, because its prompt lists up to 20 candidates; the rewrite's mean grew too, while its median stayed at 0.31 s. For one user the two effects almost cancel out: the median is 3.9 s instead of 3.6 s, and the p95 12.9 s instead of 14.9 s.

With four requests in flight (U1 against A2) the throughput is unchanged within the run-to-run noise (12.0 against 12.2 per minute), but the median grows by 16 % (20.4 s against 17.6 s) and the p95 by 12 % (39.7 s against 35.4 s), more than that noise. The longer grading prompts wait in the same queue as everything else: under load `grade_documents` takes 3.95 s instead of 2.67 s and `rewrite_query` 5.06 s instead of 3.92 s. The bottleneck is the same, one decoding slot, and the update moves work from the tool path to the search path.

## Optimization proposals

### 1. Parallel decoding slots, with a model that batches (measured)

`OLLAMA_NUM_PARALLEL=4` (here with `OLLAMA_FLASH_ATTENTION=1`) lets Ollama decode up to four sequences together. For Qwen2.5-7B this doubles the throughput of the whole workflow and halves the tail latency (C1 against A1):

| Qwen2.5-7B, 4 concurrent requests | Throughput | p50 | p95 |
|---|---|---|---|
| 1 slot (A1) | 9.5 / min | 20.2 s | 58.0 s |
| 4 slots (C1) | 20.0 / min (+111 %) | 10.3 s (−49 %) | 25.0 s (−57 %) |

It does nothing for Qwen3.5-4B. A micro-benchmark (measured by hand: 8 identical requests of about 150 generated tokens, 4 at a time, clean server) shows why:

| Model | 1 slot | 4 slots |
|---|---|---|
| Qwen2.5-7B | 59 tokens/s | 181 tokens/s |
| Qwen3.5-4B | 85 tokens/s | 88 tokens/s |

Qwen3.5's hybrid architecture is not batched by this Ollama version, so its sequences are still decoded one after the other. The slots cost memory: Qwen2.5-7B with four 8 192-token slots needs 6.8 GB and spilled 8 % of its layers to the CPU on the 8 GB GPU. **Proposal:** for several concurrent users, serve a transformer model with parallel slots (or a server with continuous batching, such as vLLM, on a GPU with more memory); for one local user, as in this prototype, the setting does not matter.

### 2. Fewer and shorter LLM calls on the critical path

Writing the answer is half of the service time (47 %), and the verification with its re-plans adds about a fifth. Measures that cut what the user waits for:

- **Stream the answer.** `synthesize_answer` produces the answer token by token; streaming it to the UI shows the first words after about the routing and the search, instead of after the whole answer and its verification (the UI already streams the steps).
- **Skip the verification where it adds nothing.** A tool answer repeats a deterministic tool output, which `finalize_response` shows verbatim anyway; verifying it costs a call (and, when the verifier errs, a re-plan) without adding safety. Re-plans happen in 14–38 % of the requests, depending on the model. *Built in the RAG reliability update:* an exact single tool result is now the answer, without the answer and verification calls; measured in U2, 0.86 verifications per request instead of 1.14, and the tool questions of the evaluation answer in about 7 s instead of 9.5 s.
- **Size the candidate pool.** The update's 20 candidates make the grading call 0.22 s longer, and under load the median grows by 16 % (U1). A pool between 4 and 20 may keep the retrieval gain at a lower cost; with 4 and the keyword ranking off, the evaluation still misses the page that the hybrid search finds. Not measured yet: compare `RETRIEVAL_CANDIDATES` 8 and 12 on hit@4 and on U1's latency.
- **Do not drop the relevance grading.** Turning it off (C2, `GRADE_WITH_LLM=false`) saves one short call per search (0.26 s), but more chunks reach the answer prompt, and the run was slower, not faster: 10.7 instead of 12.2 requests per minute, p95 41 s instead of 35 s. The grading pays for itself.

## Consequence for the model choice

The load test settles decision 4. With the server's defaults, Qwen3.5-4B (thinking off) serves 28 % more requests per minute than Qwen2.5-7B with four concurrent users (12.2 against 9.5) and cuts the p95 by 39 % (35 s against 58 s), because each call is faster and it re-plans less (1.15 answers per request against 1.38); for one user its median is 3.6 s. Together with the better evaluation scores ([evaluation.md](evaluation.md#model-comparison)), it became the default: `OLLAMA_MODEL=qwen3.5:4b`, `OLLAMA_REASONING=false`. The trade-off is scaling: with parallel slots, the 7B transformer serves more concurrent users (C1), so a multi-user deployment should reconsider the model together with the serving stack.

## Limitations

- One machine, one run per configuration; differences of about 10 % between runs of the same configuration are possible, as the evaluation's repeated runs showed. U1 and U2 ran a day after the other rows, on the same host, server settings, model builds and corpus snapshot.
- The load comes from one process on the same machine as Ollama, and the embedding model shares the CPU with it.
- The question mix is the evaluation set, cycled; real traffic would have other proportions of searches, tools and re-plans.
- A first attempt with four slots for Qwen3.5-4B was stopped after 20 minutes without a report, and a run without grading on four slots overlapped the micro-benchmarks; neither is reported. During long runs `ollama ps` shows the model as *Stopping...* while it keeps serving requests normally.

## Reproducing the runs

With the corpus and the index built and Ollama serving the models on the host:

```bash
export LLM_PROVIDER=ollama EMBEDDING_PROVIDER=huggingface
LLM_PROVIDER=fake EMBEDDING_PROVIDER=fake uv run agentic-rag loadtest          # framework baseline (fake index)
uv run agentic-rag loadtest --requests 100 --concurrency 4                     # the default model, Qwen3.5-4B
uv run agentic-rag loadtest --requests 50 --concurrency 1                      # service times without queueing
OLLAMA_MODEL=qwen2.5:7b-instruct uv run agentic-rag loadtest --requests 100 --concurrency 4
GRADE_WITH_LLM=false uv run agentic-rag loadtest --requests 100 --concurrency 4
```

U1 and U2 are the second and third commands with the RAG reliability update and its default settings. For C1, restart Ollama with `OLLAMA_NUM_PARALLEL=4 OLLAMA_FLASH_ATTENTION=1 ollama serve` before the Qwen2.5 run. Each run writes `loadtest-<provider>-c<concurrency>-<UTC time>.json` and its Markdown summary to `data/eval/results/` and prints the summary.
