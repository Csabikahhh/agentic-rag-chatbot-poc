# Evaluation of the node `analyze_request`

- Run: 2026-10-03 14:19 UTC, dataset `data/eval/questions.jsonl`, 17 questions, 0 failed
- LLM: ollama `hf.co/bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M`; judge: none
- Embeddings: huggingface `intfloat/multilingual-e5-small`; TOP_K=4; GRADE_WITH_LLM=true

| Metric | Score | Questions |
|---|---|---|
| Routing accuracy | 0.88 | 16 |
| Retrieval hit@4 | – | 0 |
| Answer correctness | – | 0 |
| Faithfulness | – | 0 |

Latency per question: mean 1.1 s, max 3.2 s.

## By tag

| Tag | Questions | Routing | hit@4 | Correctness | Faithfulness |
|---|---|---|---|---|---|
| browser-support | 2 | 1.00 | – | – | – |
| contrast | 2 | 0.50 | – | – | – |
| english | 12 | 0.91 | – | – | – |
| hungarian | 5 | 0.80 | – | – | – |
| multi-part | 3 | 0.67 | – | – | – |
| nextjs | 3 | 1.00 | – | – | – |
| nuxt | 3 | 1.00 | – | – | – |
| react | 3 | 1.00 | – | – | – |
| single-hop | 8 | 1.00 | – | – | – |
| tool | 5 | 0.60 | – | – | – |

## Questions

| Id | Tags | Route (expected → chosen) | hit@4 | Correctness | Faithfulness | Latency |
|---|---|---|---|---|---|---|
| q01 | single-hop, english, mdn | single → single ✓ | – | – | – | 3.2 s |
| q02 | single-hop, english, react | single → single ✓ | – | – | – | 0.7 s |
| q03 | single-hop, english, vue | single → single ✓ | – | – | – | 0.8 s |
| q04 | single-hop, english, nextjs | single → single ✓ | – | – | – | 0.9 s |
| q05 | single-hop, english, nuxt | single → single ✓ | – | – | – | 0.8 s |
| q06 | single-hop, english, typescript | single → single ✓ | – | – | – | 0.7 s |
| q07 | single-hop, hungarian, nextjs | single → single ✓ | – | – | – | 0.9 s |
| q08 | single-hop, hungarian, react | single → single ✓ | – | – | – | 0.7 s |
| q09 | multi-part, english, nextjs, nuxt | complex → complex ✓ | – | – | – | 0.7 s |
| q10 | multi-part, hungarian, react, nuxt | complex → complex ✓ | – | – | – | 0.7 s |
| q11 | tool, english, contrast | tool → tool ✓ | – | – | – | 1.3 s |
| q12 | tool, english, specificity | tool → single ✗ | – | – | – | 0.9 s |
| q13 | tool, english, browser-support | tool → tool ✓ | – | – | – | 1.1 s |
| q14 | tool, hungarian, browser-support | tool → tool ✓ | – | – | – | 1.4 s |
| q15 | multi-part, tool, hungarian, contrast | complex → tool ✗ | – | – | – | 2.1 s |
| q16 | direct, english | direct → direct ✓ | – | – | – | 1.2 s |
| q17 | out-of-scope, english | – → single | – | – | – | 0.6 s |
