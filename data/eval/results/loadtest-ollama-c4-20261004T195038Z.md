# Load test: 100 requests, 4 at a time

- Run: 2026-10-04 19:50 UTC, 3 warm-up requests before the measured ones
- LLM: ollama `qwen3.5:4b`, OLLAMA_REASONING=false; embeddings: huggingface; TOP_K=4; GRADE_WITH_LLM=true
- Duration 498.7 s, throughput 0.201 requests/s (12.0 per minute), errors 0 (0%)

| Latency | Mean | Min | p50 | p95 | p99 | Max |
|---|---|---|---|---|---|---|
| End to end (100 requests) | 19.76 s | 1.94 s | 20.41 s | 39.71 s | 42.70 s | 66.50 s |
| Warm-up (3) | 16.44 s | 4.30 s | 15.46 s | 28.15 s | 29.28 s | 29.56 s |

## Per node

| Node | Runs per request | Mean | p50 | p95 | Max | Share of request time |
|---|---|---|---|---|---|---|
| `run_rag_subtask` | 0.83 | 9.03 s | 8.58 s | 14.69 s | 16.75 s | 38% |
| `synthesize_answer` | 1.11 | 4.72 s | 4.44 s | 10.67 s | 15.83 s | 27% |
| ↳ `rewrite_query` | 0.83 | 5.06 s | 4.49 s | 9.55 s | 10.38 s | 21% |
| `analyze_request` | 1.00 | 3.95 s | 3.34 s | 7.72 s | 8.54 s | 20% |
| ↳ `grade_documents` | 0.83 | 3.95 s | 3.26 s | 7.93 s | 8.79 s | 17% |
| `verify_answer` | 0.87 | 3.43 s | 3.00 s | 7.22 s | 11.96 s | 15% |
| `plan_subtasks` | 0.35 | 4.12 s | 3.49 s | 7.63 s | 8.57 s | 7% |
| ↳ `retrieve` | 0.83 | 0.02 s | 0.02 s | 0.02 s | 0.03 s | 0% |
| `call_tool` | 0.48 | 0.02 s | 0.00 s | 0.00 s | 0.93 s | 0% |
| `finalize_response` | 1.00 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0% |
| ↳ `build_context` | 0.83 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0% |

↳ marks the RAG subgraph's nodes: their time is part of `run_rag_subtask`'s. Parallel workers overlap, so the shares do not add up to 100 %.
