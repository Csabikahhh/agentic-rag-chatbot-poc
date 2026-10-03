# Load test: 100 requests, 4 at a time

- Run: 2026-10-03 15:59 UTC, 3 warm-up requests before the measured ones
- LLM: ollama `hf.co/bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M`; embeddings: huggingface; TOP_K=4; GRADE_WITH_LLM=true
- Duration 629.2 s, throughput 0.159 requests/s (9.5 per minute), errors 0 (0%)

| Latency | Mean | Min | p50 | p95 | p99 | Max |
|---|---|---|---|---|---|---|
| End to end (100 requests) | 25.09 s | 2.45 s | 20.21 s | 58.00 s | 65.23 s | 75.01 s |
| Warm-up (3) | 13.02 s | 6.57 s | 6.92 s | 23.69 s | 25.18 s | 25.56 s |

## Per node

| Node | Runs per request | Mean | p50 | p95 | Max | Share of request time |
|---|---|---|---|---|---|---|
| `synthesize_answer` | 1.38 | 6.28 s | 5.31 s | 13.68 s | 17.83 s | 35% |
| `run_rag_subtask` | 1.02 | 7.78 s | 7.23 s | 17.10 s | 18.82 s | 32% |
| ↳ `rewrite_query` | 1.02 | 5.17 s | 4.10 s | 13.44 s | 16.86 s | 21% |
| `verify_answer` | 1.38 | 3.71 s | 3.14 s | 9.21 s | 13.79 s | 20% |
| `analyze_request` | 1.00 | 2.72 s | 2.13 s | 6.19 s | 9.14 s | 11% |
| ↳ `grade_documents` | 1.02 | 2.58 s | 1.22 s | 8.60 s | 9.68 s | 10% |
| `plan_subtasks` | 0.56 | 3.73 s | 2.68 s | 10.30 s | 13.73 s | 8% |
| ↳ `retrieve` | 1.02 | 0.02 s | 0.02 s | 0.03 s | 0.03 s | 0% |
| `call_tool` | 0.60 | 0.01 s | 0.00 s | 0.00 s | 0.25 s | 0% |
| `finalize_response` | 1.00 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0% |
| ↳ `build_context` | 1.02 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0% |

↳ marks the RAG subgraph's nodes: their time is part of `run_rag_subtask`'s. Parallel workers overlap, so the shares do not add up to 100 %.
