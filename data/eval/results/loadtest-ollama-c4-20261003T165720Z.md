# Load test: 100 requests, 4 at a time

- Run: 2026-10-03 16:57 UTC, 3 warm-up requests before the measured ones
- LLM: ollama `hf.co/bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M`, OLLAMA_REASONING=false; embeddings: huggingface; TOP_K=4; GRADE_WITH_LLM=true
- Duration 300.1 s, throughput 0.333 requests/s (20.0 per minute), errors 0 (0%)

| Latency | Mean | Min | p50 | p95 | p99 | Max |
|---|---|---|---|---|---|---|
| End to end (100 requests) | 11.92 s | 1.37 s | 10.25 s | 24.95 s | 42.53 s | 62.85 s |
| Warm-up (3) | 20.46 s | 6.92 s | 8.18 s | 42.47 s | 45.52 s | 46.28 s |

## Per node

| Node | Runs per request | Mean | p50 | p95 | Max | Share of request time |
|---|---|---|---|---|---|---|
| `synthesize_answer` | 1.26 | 5.88 s | 4.84 s | 14.43 s | 19.30 s | 62% |
| `analyze_request` | 1.00 | 1.55 s | 1.31 s | 3.40 s | 4.55 s | 13% |
| `run_rag_subtask` | 0.96 | 1.55 s | 0.81 s | 3.23 s | 3.95 s | 13% |
| `verify_answer` | 1.26 | 0.86 s | 0.56 s | 2.08 s | 2.98 s | 9% |
| ↳ `rewrite_query` | 0.96 | 1.11 s | 0.35 s | 2.48 s | 2.76 s | 9% |
| `plan_subtasks` | 0.44 | 1.40 s | 1.24 s | 2.43 s | 3.13 s | 5% |
| ↳ `grade_documents` | 0.96 | 0.42 s | 0.41 s | 0.77 s | 1.17 s | 3% |
| ↳ `retrieve` | 0.96 | 0.03 s | 0.02 s | 0.04 s | 0.17 s | 0% |
| `call_tool` | 0.53 | 0.04 s | 0.00 s | 0.00 s | 2.22 s | 0% |
| `finalize_response` | 1.00 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0% |
| ↳ `build_context` | 0.96 | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0% |

↳ marks the RAG subgraph's nodes: their time is part of `run_rag_subtask`'s. Parallel workers overlap, so the shares do not add up to 100 %.
