# Evaluation of the node `run_rag_subtask`

- Run: 2026-10-04 19:12 UTC, dataset `data/eval/questions.jsonl`, 17 questions, 0 failed
- LLM: ollama `qwen3.5:4b`; judge: none
- Embeddings: huggingface `intfloat/multilingual-e5-small`; TOP_K=4; GRADE_WITH_LLM=true

| Metric | Score | Questions |
|---|---|---|
| Routing accuracy | – | 0 |
| Retrieval hit@4 | 1.00 | 10 |
| Complete evidence@4 | 0.50 | 2 |
| Answer correctness | – | 0 |
| Faithfulness | – | 0 |

Latency per question: mean 2.4 s, max 17.6 s.

## By tag

| Tag | Questions | Routing | hit@4 | Correctness | Faithfulness |
|---|---|---|---|---|---|
| browser-support | 2 | – | – | – | – |
| contrast | 2 | – | – | – | – |
| english | 12 | – | 1.00 | – | – |
| hungarian | 5 | – | 1.00 | – | – |
| multi-part | 3 | – | 1.00 | – | – |
| nextjs | 3 | – | 1.00 | – | – |
| nuxt | 3 | – | 1.00 | – | – |
| react | 3 | – | 1.00 | – | – |
| single-hop | 8 | – | 1.00 | – | – |
| tool | 5 | – | – | – | – |

## Questions

| Id | Tags | Route (expected → chosen) | hit@4 | Correctness | Faithfulness | Latency |
|---|---|---|---|---|---|---|
| q01 | single-hop, english, mdn | single → – | ✓ | – | – | 17.6 s |
| q02 | single-hop, english, react | single → – | ✓ | – | – | 1.3 s |
| q03 | single-hop, english, vue | single → – | ✓ | – | – | 1.2 s |
| q04 | single-hop, english, nextjs | single → – | ✓ | – | – | 1.5 s |
| q05 | single-hop, english, nuxt | single → – | ✓ | – | – | 1.1 s |
| q06 | single-hop, english, typescript | single → – | ✓ | – | – | 2.0 s |
| q07 | single-hop, hungarian, nextjs | single → – | ✓ | – | – | 1.5 s |
| q08 | single-hop, hungarian, react | single → – | ✓ | – | – | 1.3 s |
| q09 | multi-part, english, nextjs, nuxt | complex → – | ✓ | – | – | 1.6 s |
| q10 | multi-part, hungarian, react, nuxt | complex → – | ✓ | – | – | 1.3 s |
| q11 | tool, english, contrast | tool → – | – | – | – | 2.0 s |
| q12 | tool, english, specificity | tool → – | – | – | – | 1.6 s |
| q13 | tool, english, browser-support | tool → – | – | – | – | 1.3 s |
| q14 | tool, hungarian, browser-support | tool → – | – | – | – | 1.5 s |
| q15 | multi-part, tool, hungarian, contrast | complex → – | – | – | – | 2.1 s |
| q16 | direct, english | direct → – | – | – | – | 1.3 s |
| q17 | out-of-scope, english | – → – | – | – | – | 1.1 s |

## Complete evidence

Every required evidence group must have a hit in a sub-task's top 4.

- **q09**: ✓
- **q10**: ✗
