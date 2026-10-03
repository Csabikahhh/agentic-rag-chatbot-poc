# Load test: 100 requests, 4 at a time

- Run: 2026-10-03 17:08 UTC, 3 warm-up requests before the measured ones
- LLM: ollama `qwen3.5:4b`, OLLAMA_REASONING=false; embeddings: huggingface; TOP_K=4; GRADE_WITH_LLM=false
- Duration 561.7 s, throughput 0.178 requests/s (10.7 per minute), errors 0 (0%)

| Latency | Mean | Min | p50 | p95 | p99 | Max |
|---|---|---|---|---|---|---|
| End to end (100 requests) | 22.16 s | 2.83 s | 18.16 s | 41.42 s | 64.34 s | 65.54 s |
| Warm-up (3) | 20.14 s | 2.86 s | 21.40 s | 34.69 s | 35.87 s | 36.17 s |

## Per node

| Node | Runs per request | Mean | p50 | p95 | Max | Share of request time |
|---|---|---|---|---|---|---|
| `synthesize_answer` | 1.21 | 5.80 s | 5.26 s | 11.25 s | 13.81 s | 32% |
| `run_rag_subtask` | 0.92 | 5.25 s | 4.74 s | 8.95 s | 12.51 s | 22% |
| ↳ `rewrite_query` | 0.92 | 5.23 s | 4.72 s | 8.93 s | 12.49 s | 22% |
| `verify_answer` | 1.21 | 3.91 s | 3.54 s | 7.52 s | 10.65 s | 21% |
| `analyze_request` | 1.00 | 4.51 s | 4.03 s | 8.76 s | 11.31 s | 20% |
| `plan_subtasks` | 0.45 | 4.04 s | 3.42 s | 9.87 s | 10.85 s | 8% |
| ↳ `retrieve` | 0.92 | 0.02 s | 0.02 s | 0.03 s | 0.09 s | 0% |
| `call_tool` | 0.48 | 0.03 s | 0.00 s | 0.00 s | 1.21 s | 0% |
| `finalize_response` | 1.00 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0% |
| ↳ `build_context` | 0.92 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0% |
| ↳ `grade_documents` | 0.92 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0% |

↳ marks the RAG subgraph's nodes: their time is part of `run_rag_subtask`'s. Parallel workers overlap, so the shares do not add up to 100 %.
