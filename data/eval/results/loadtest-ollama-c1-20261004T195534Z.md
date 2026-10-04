# Load test: 50 requests, 1 at a time

- Run: 2026-10-04 19:55 UTC, 3 warm-up requests before the measured ones
- LLM: ollama `qwen3.5:4b`, OLLAMA_REASONING=false; embeddings: huggingface; TOP_K=4; GRADE_WITH_LLM=true
- Duration 256.4 s, throughput 0.195 requests/s (11.7 per minute), errors 0 (0%)

| Latency | Mean | Min | p50 | p95 | p99 | Max |
|---|---|---|---|---|---|---|
| End to end (50 requests) | 5.13 s | 0.91 s | 3.94 s | 12.94 s | 15.01 s | 15.05 s |
| Warm-up (3) | 12.36 s | 2.96 s | 10.09 s | 22.64 s | 23.76 s | 24.04 s |

## Per node

| Node | Runs per request | Mean | p50 | p95 | Max | Share of request time |
|---|---|---|---|---|---|---|
| `synthesize_answer` | 1.10 | 2.25 s | 2.22 s | 5.20 s | 5.31 s | 48% |
| `run_rag_subtask` | 0.80 | 1.71 s | 1.13 s | 3.24 s | 3.31 s | 27% |
| ↳ `rewrite_query` | 0.80 | 1.21 s | 0.31 s | 2.42 s | 2.47 s | 19% |
| `analyze_request` | 1.00 | 0.88 s | 0.82 s | 1.13 s | 1.15 s | 17% |
| `verify_answer` | 0.86 | 0.46 s | 0.26 s | 1.37 s | 1.68 s | 8% |
| ↳ `grade_documents` | 0.80 | 0.48 s | 0.48 s | 0.85 s | 0.91 s | 8% |
| `plan_subtasks` | 0.34 | 1.01 s | 1.20 s | 1.24 s | 1.24 s | 7% |
| ↳ `retrieve` | 0.80 | 0.02 s | 0.02 s | 0.02 s | 0.03 s | 0% |
| `call_tool` | 0.48 | 0.02 s | 0.00 s | 0.00 s | 0.36 s | 0% |
| `finalize_response` | 1.00 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0% |
| ↳ `build_context` | 0.80 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0% |

↳ marks the RAG subgraph's nodes: their time is part of `run_rag_subtask`'s. Parallel workers overlap, so the shares do not add up to 100 %.
