# Load test: 100 requests, 4 at a time

- Run: 2026-10-03 16:08 UTC, 3 warm-up requests before the measured ones
- LLM: ollama `qwen3.5:4b`, OLLAMA_REASONING=false; embeddings: huggingface; TOP_K=4; GRADE_WITH_LLM=true
- Duration 493.2 s, throughput 0.203 requests/s (12.2 per minute), errors 0 (0%)

| Latency | Mean | Min | p50 | p95 | p99 | Max |
|---|---|---|---|---|---|---|
| End to end (100 requests) | 19.44 s | 3.17 s | 17.58 s | 35.35 s | 50.08 s | 53.17 s |
| Warm-up (3) | 15.49 s | 2.46 s | 15.34 s | 27.34 s | 28.41 s | 28.68 s |

## Per node

| Node | Runs per request | Mean | p50 | p95 | Max | Share of request time |
|---|---|---|---|---|---|---|
| `synthesize_answer` | 1.15 | 5.57 s | 4.65 s | 9.78 s | 13.44 s | 33% |
| `run_rag_subtask` | 0.86 | 6.61 s | 6.62 s | 10.50 s | 15.66 s | 29% |
| `analyze_request` | 1.00 | 3.48 s | 3.32 s | 5.84 s | 8.06 s | 18% |
| `verify_answer` | 1.15 | 3.01 s | 2.66 s | 6.38 s | 8.14 s | 18% |
| ↳ `rewrite_query` | 0.86 | 3.92 s | 3.71 s | 7.74 s | 11.01 s | 17% |
| ↳ `grade_documents` | 0.86 | 2.67 s | 2.60 s | 5.32 s | 6.13 s | 12% |
| `plan_subtasks` | 0.39 | 3.34 s | 2.62 s | 4.97 s | 9.95 s | 7% |
| `call_tool` | 0.48 | 0.03 s | 0.00 s | 0.00 s | 1.29 s | 0% |
| ↳ `retrieve` | 0.86 | 0.02 s | 0.02 s | 0.02 s | 0.05 s | 0% |
| `finalize_response` | 1.00 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0% |
| ↳ `build_context` | 0.86 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0% |

↳ marks the RAG subgraph's nodes: their time is part of `run_rag_subtask`'s. Parallel workers overlap, so the shares do not add up to 100 %.
