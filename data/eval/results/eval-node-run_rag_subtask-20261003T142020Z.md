# Evaluation of the node `run_rag_subtask`

- Run: 2026-10-03 14:20 UTC, dataset `data/eval/questions.jsonl`, 17 questions, 0 failed
- LLM: ollama `hf.co/bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M`; judge: none
- Embeddings: huggingface `intfloat/multilingual-e5-small`; TOP_K=4; GRADE_WITH_LLM=true

| Metric | Score | Questions |
|---|---|---|
| Routing accuracy | – | 0 |
| Retrieval hit@4 | 0.80 | 10 |
| Answer correctness | – | 0 |
| Faithfulness | – | 0 |

Latency per question: mean 1.4 s, max 16.2 s.

## By tag

| Tag | Questions | Routing | hit@4 | Correctness | Faithfulness |
|---|---|---|---|---|---|
| browser-support | 2 | – | – | – | – |
| contrast | 2 | – | – | – | – |
| english | 12 | – | 0.71 | – | – |
| hungarian | 5 | – | 1.00 | – | – |
| multi-part | 3 | – | 1.00 | – | – |
| nextjs | 3 | – | 1.00 | – | – |
| nuxt | 3 | – | 1.00 | – | – |
| react | 3 | – | 0.67 | – | – |
| single-hop | 8 | – | 0.75 | – | – |
| tool | 5 | – | – | – | – |

## Questions

| Id | Tags | Route (expected → chosen) | hit@4 | Correctness | Faithfulness | Latency |
|---|---|---|---|---|---|---|
| q01 | single-hop, english, mdn | single → – | ✓ | – | – | 16.2 s |
| q02 | single-hop, english, react | single → – | ✗ | – | – | 0.6 s |
| q03 | single-hop, english, vue | single → – | ✗ | – | – | 0.3 s |
| q04 | single-hop, english, nextjs | single → – | ✓ | – | – | 0.5 s |
| q05 | single-hop, english, nuxt | single → – | ✓ | – | – | 0.5 s |
| q06 | single-hop, english, typescript | single → – | ✓ | – | – | 0.5 s |
| q07 | single-hop, hungarian, nextjs | single → – | ✓ | – | – | 0.6 s |
| q08 | single-hop, hungarian, react | single → – | ✓ | – | – | 0.5 s |
| q09 | multi-part, english, nextjs, nuxt | complex → – | ✓ | – | – | 0.9 s |
| q10 | multi-part, hungarian, react, nuxt | complex → – | ✓ | – | – | 0.4 s |
| q11 | tool, english, contrast | tool → – | – | – | – | 0.7 s |
| q12 | tool, english, specificity | tool → – | – | – | – | 0.7 s |
| q13 | tool, english, browser-support | tool → – | – | – | – | 0.4 s |
| q14 | tool, hungarian, browser-support | tool → – | – | – | – | 0.4 s |
| q15 | multi-part, tool, hungarian, contrast | complex → – | – | – | – | 1.0 s |
| q16 | direct, english | direct → – | – | – | – | 0.4 s |
| q17 | out-of-scope, english | – → – | – | – | – | 0.1 s |
