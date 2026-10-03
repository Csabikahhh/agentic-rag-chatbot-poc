# Load test: 50 requests, 1 at a time

- Run: 2026-10-03 16:13 UTC, 3 warm-up requests before the measured ones
- LLM: ollama `qwen3.5:4b`, OLLAMA_REASONING=false; embeddings: huggingface; TOP_K=4; GRADE_WITH_LLM=true
- Duration 261.4 s, throughput 0.191 requests/s (11.5 per minute), errors 0 (0%)

| Latency | Mean | Min | p50 | p95 | p99 | Max |
|---|---|---|---|---|---|---|
| End to end (50 requests) | 5.23 s | 1.18 s | 3.64 s | 14.91 s | 15.94 s | 16.36 s |
| Warm-up (3) | 15.67 s | 2.68 s | 13.99 s | 28.70 s | 30.01 s | 30.34 s |

## Per node

| Node | Runs per request | Mean | p50 | p95 | Max | Share of request time |
|---|---|---|---|---|---|---|
| `synthesize_answer` | 1.14 | 2.18 s | 1.90 s | 4.87 s | 5.90 s | 47% |
| `run_rag_subtask` | 0.84 | 1.15 s | 0.59 s | 2.86 s | 3.00 s | 18% |
| `analyze_request` | 1.00 | 0.94 s | 0.87 s | 1.25 s | 1.32 s | 18% |
| ↳ `rewrite_query` | 0.84 | 0.87 s | 0.31 s | 2.38 s | 2.43 s | 14% |
| `verify_answer` | 1.14 | 0.54 s | 0.31 s | 1.65 s | 1.72 s | 12% |
| `plan_subtasks` | 0.38 | 1.03 s | 0.97 s | 1.38 s | 1.41 s | 8% |
| ↳ `grade_documents` | 0.84 | 0.26 s | 0.26 s | 0.46 s | 0.56 s | 4% |
| `call_tool` | 0.48 | 0.05 s | 0.00 s | 0.00 s | 1.27 s | 0% |
| ↳ `retrieve` | 0.84 | 0.02 s | 0.02 s | 0.02 s | 0.03 s | 0% |
| `finalize_response` | 1.00 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0% |
| ↳ `build_context` | 0.84 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0% |

↳ marks the RAG subgraph's nodes: their time is part of `run_rag_subtask`'s. Parallel workers overlap, so the shares do not add up to 100 %.
