# Functional evaluation

> **Status:** Phase 7, 2026-10-03. Every number below comes from a report committed in [`data/eval/results/`](../data/eval/results/), except the first run before the fixes and the timing of one single call, and can be reproduced with the commands at the end. The load test follows in Phase 8.

## Contents

1. [Method](#method)
2. [Results](#results)
3. [Findings per question](#findings-per-question)
4. [Model comparison](#model-comparison)
5. [How reliable the judge is](#how-reliable-the-judge-is)
6. [What the evaluation changed](#what-the-evaluation-changed)
7. [Limitations](#limitations)
8. [Conclusions](#conclusions)
9. [Reproducing the runs](#reproducing-the-runs)

## Method

### Question set

[`data/eval/questions.jsonl`](../data/eval/questions.jsonl) holds 17 questions over the downloaded documentation, each with a reference answer written from the corpus, the documents that answer it and the route the router should choose:

| Group | Questions | What it checks |
|---|---|---|
| Single search | q01–q08: MDN, React (2), Vue, Next.js (2), Nuxt, TypeScript | Retrieval from each source and a cited answer; q07 and q08 in Hungarian |
| Multi-part | q09 (Next.js and Nuxt data fetching), q10 (React and Nuxt `useState`, Hungarian), q15 (two contrast checks, Hungarian) | Decomposition into parallel sub-tasks |
| Tool | q11 (contrast), q12 (specificity), q13 and q14 (browser support, q14 in Hungarian) | Tool selection and the tool's verdict |
| Direct | q16 (a greeting) | A reply without search |
| Out of scope | q17 (*What is the capital of France?*) | Declining instead of answering from general knowledge |

Five questions are in Hungarian: the corpus is English, so they test the cross-lingual path (the English query rewrite) and the Hungarian answers.

### Metrics

- **Routing accuracy:** the route `analyze_request` chose against `expected_intent` (16 questions; q17 accepts any route).
- **Retrieval hit@4:** whether a retrieve sub-task ranked a chunk of an expected document among its first `TOP_K = 4` chunks, counted after the grading step, so a relevant chunk that the grading drops is a miss (10 questions).
- **Answer correctness:** a judge model compares the answer with the reference answer in substance and picks *correct* (1), *partially correct* (0.5) or *incorrect* (0) (17 questions).
- **Faithfulness:** the judge checks every claim of the answer against the material of the run, the documentation excerpts and the tool outputs: *supported* (1), *partially supported* (0.5) or *unsupported* (0). It applies to every run that searched or called a tool, also when that produced nothing, so invented figures count; direct replies are not scored (16 questions).

The judge picks one of three verdicts instead of writing a score, because a small local model gives a verdict far more consistently than a number. `architecture.md` describes the [evaluation harness](architecture.md#evaluation).

### Setup

- Windows 11, NVIDIA RTX 5070 Laptop GPU (8 GB), Ollama 0.35.0 on the host.
- LLM: Qwen2.5-7B-Instruct, 4-bit `Q4_K_M` (the `hf.co/bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M` build, the same model and quantization as the default `qwen2.5:7b-instruct` tag), temperature 0, `OLLAMA_NUM_CTX=8192`. For the comparison: Qwen3.5-4B (`qwen3.5:4b`), with and without its thinking mode.
- Judge: the same Qwen2.5-7B build in every run, also when Qwen3.5 answers (`--judge-model`).
- Embeddings `intfloat/multilingual-e5-small` on the CPU, the full index (18 654 chunks), `TOP_K=4`, `GRADE_WITH_LLM=true`, `MAX_RETRIES=2`.

## Results

Three runs of the full graph per configuration, because the runs vary (see [Limitations](#limitations)); the table gives the mean and, in brackets, the range:

| Model | Routing | hit@4 | Correctness | Faithfulness | Latency per question (median) |
|---|---|---|---|---|---|
| Qwen2.5-7B (default) | 0.88 | 0.90 | 0.88 (0.82–0.91) | 0.91 (0.88–0.94) | 8.4 s |
| Qwen3.5-4B, thinking off | 1.00 | 0.90 | 0.91 | 0.94 | 12 s ¹ |
| Qwen3.5-4B, thinking on (one run) | 1.00 | 0.90 | 0.91 | 0.94 | 143 s |

¹ Inflated: the two models do not fit in the 8 GB of VRAM together, so Ollama reloads Qwen3.5 after every judge call. The node runs below have no judge.

The node targets measure one step alone, without a judge (one run each):

| Model | Routing alone (`analyze_request`) | Median call | Retrieval alone (`run_rag_subtask`), hit@4 | Median call |
|---|---|---|---|---|
| Qwen2.5-7B | 0.88 | 0.9 s | 0.80 | 0.5 s |
| Qwen3.5-4B, thinking off | 1.00 | 0.8 s | 0.90 | 0.6 s |

By language and kind of question, mean of the three runs per model:

| Group | Qwen2.5-7B correctness | Qwen2.5-7B faithfulness | Qwen3.5-4B correctness | Qwen3.5-4B faithfulness |
|---|---|---|---|---|
| English (12) | 0.92 | 0.97 | 0.92 | 0.91 |
| Hungarian (5) | 0.80 | 0.77 | 0.90 | 1.00 |
| Single search (8) | 0.85 | 0.98 | 0.81 | 0.88 |
| Multi-part (3) | 0.78 | 0.61 | 1.00 | 1.00 |
| Tool (5) | 0.90 | 0.80 | 1.00 | 1.00 |

The question-by-question tables are in the Markdown summaries next to every report, for example [`eval-graph-20261003T141916Z.md`](../data/eval/results/eval-graph-20261003T141916Z.md).

## Findings per question

With Qwen2.5-7B, the default:

- **Retrieval works for both languages.** hit@4 is 0.90 on the graph: the English query rewrite brings the Hungarian questions (q07, q08, q10) to the right pages. The only miss is q03: for *computed property with the Composition API* the search returns the Options API reference (`vue/api/options-state.md`) or the Composition API FAQ, never the guide page `computed.md`. Retrieval alone (one search for the whole question) scores 0.80: the graph also recovers q02 through its plan.
- **Routing errs on two tool questions, every time.** The specificity question (q12) goes to a search, and the two-contrast question (q15) to a single tool call. The verification and the re-plan repair q12 in every run (the answer is correct); q15 is answered correctly in two runs out of three, and in the third, one ratio is invented (*4.5:1* for black on `#f5f5f5`, really 19.26:1).
- **The weak spot is Hungarian.** The Hungarian answers score 0.80 for correctness and 0.77 for faithfulness, against 0.92 and 0.97 in English. The multi-part Hungarian question (q10) mixes the two frameworks' terms (*both are Hooks*) in two runs of three, and q15 is the worst question of the set.
- **Out of scope and tools are reliable.** q17 is declined in every run, q11, q13 and q14 give the tool's verdict, q16 greets without a search.

## Model comparison

Qwen3.5-4B with its thinking mode switched off (`OLLAMA_REASONING=false`) answers better than the 7B default on every metric that differs: it routes all 16 questions right (q12 and q15 included, in all three node and graph runs), retrieves better alone (0.90 against 0.80, through a better English rewrite), and its Hungarian and multi-part answers are correct and faithful in every run. It gave the same scores in all three runs, while the 7B model varied. Its one systematic failure is q03: it follows the retrieved Options API reference faithfully and gets the Composition API wrong (correctness 0), where the 7B model answers partly from general knowledge.

With thinking on, the quality is the same, but every call writes a long hidden reasoning first: a question took 143 s at the median, against 12 s with thinking off. One call measured on its own: 13.8 s with thinking, 0.7 s without. Thinking is therefore useless for this agent, which makes four to eight LLM calls per question.

The model is 3.4 GB instead of 4.7 GB, so it leaves more of an 8 GB GPU for the KV cache of parallel requests. Per call it is as fast as the 7B model (0.8 s against 0.9 s routing, 0.6 s against 0.5 s retrieval); the end-to-end latency without the judge's model swaps is what the load test measures.

## How reliable the judge is

A manual review of the 17 answers of the first committed run agrees with 15 of the judge's correctness verdicts:

- **q02** (*correct*, judged partially correct): the answer shows a class component before `useState`; it is correct, and the judge's rationale invents a contradiction about React versions.
- **q08** (*correct*, judged partially correct): the judge reads the documented order of cleanup and setup as an error.

The faithfulness verdicts are mostly right but lenient once: in q03 the answer's `computed` example is not in the retrieved material, and the judge still rates it supported. The judge did catch the invented ratio of q15 and the wrong API of q03 (Qwen3.5). On these numbers the judge is slightly too strict on correctness, so the correctness scores are, if anything, a little low; the comparison between the models is not affected, because the same judge scores both.

## What the evaluation changed

The first run (not committed: it ran before the fixes; routing 0.88, hit@4 0.70, correctness 0.79, faithfulness 0.89) found defects in the system and in the measurement. They were fixed before the committed runs:

| Finding | Change |
|---|---|
| *What is the capital of France?* got the answer *Paris*: the router writes the reply of the `direct` route itself, and the 7B model answered from general knowledge, although the prompt showed this very question with a refusal | The `direct` route is now only for greetings, thanks and questions about the assistant. A question outside the topic goes to `single`: the search finds nothing, and the answer says that the documentation does not cover it, so the corpus decides instead of the model's obedience (the prompt's example is now a different question) |
| The two-contrast question (q15) was routed to one `check_contrast` call with invented arguments (`foreground2`), the call failed, and the answer invented both ratios. Faithfulness did not see it, because it was only judged when there was material | The router prompt says that a question needing the same tool twice is `complex`, with an example; faithfulness is now judged for every run that searched or called a tool, also without material |
| The specificity question was routed to a search | The router prompt got a `css_specificity` example (not taken from the question set); the 7B model still routes q12 to a search |
| A tool result kept from a rejected round was labelled `tool` instead of its name, and a check that a re-plan repeated was shown twice | `SubtaskResult.tool_name` names the tool; `finalize_response` shows each tool output once |
| The judge rated *Paris* correct for a question whose reference answer is a refusal, and wrong ratios only partially correct | The correctness prompt says that a declining answer is correct when the reference says so, and that a wrong number, version or verdict on the main point makes an answer incorrect |
| q07 and q09 retrieved the Next.js `fetch` API reference and the Nuxt `$fetch` page, which answer the questions as well as the expected guide pages | Those pages (and the Nuxt `useFetch` and `useAsyncData` pages) were added to `expected_documents`; the items' `notes` record it |
| Qwen3.5-4B took minutes per question | `OLLAMA_REASONING` turns the thinking mode of reasoning models off (`false`) or on; unset keeps the model's default |

The router examples are not questions of the set, and the expected documents were only extended with pages that contain the answer, so the set still measures generalization rather than the prompt.

## Limitations

- **Seventeen questions.** One question moves a metric by about 0.06 (by 0.1 to 0.2 in a group of five), so differences below 0.1 are noise.
- **Run-to-run variance.** Even at temperature 0 the 7B runs differ (correctness 0.82 to 0.91): the parallel searches of a complex question reach Ollama at the same time, and a different first draft changes the verifier's decision and the re-plan. Qwen3.5 gave identical scores in all three runs.
- **Self-judging.** The 7B model judges its own answers; only local models are allowed, and none is clearly stronger. The [manual review](#how-reliable-the-judge-is) shows where the judge errs, and the model comparison keeps the judge fixed.
- **Latency.** Measured one question at a time on one machine; the first question includes loading the models (about 15 s), and the Qwen3.5 graph runs include the judge's model swaps. The load test measures latency properly.
- **No conversation context.** Every question runs on a fresh state; follow-up questions are not evaluated.

## Conclusions

1. **The agentic workflow does what it is built for.** With the default 7B model the answers score 0.88 for correctness and 0.91 for faithfulness on average; retrieval finds the right page for 9 of 10 questions in both languages, the out-of-scope question is declined, and the tools deliver exact verdicts that are always shown verbatim.
2. **The weak points are routing tool questions and Hungarian.** The 7B router misroutes two of the five tool questions; the verification and the re-plan repair q12 in every run and q15 in two runs of three. Hungarian answers lose about 0.12 in correctness and 0.20 in faithfulness against English.
3. **Decision 4 should change: Qwen3.5-4B with thinking off is the better model here.** It fixes both weak points (routing 1.00, Hungarian 0.90 and 1.00), is stable from run to run and smaller. It must run with `OLLAMA_REASONING=false`: with thinking on it is about ten times slower without being better. The switch of the default is left for the load test (Phase 8), which compares the two models' latency under load before the default changes.
4. **Retrieval has one known gap:** the Vue guide page on computed properties is outranked by the API reference. A larger embedding model, or a hybrid keyword search, is the next lever; with `TOP_K=4` and grading on, hit@4 is 0.90.

## Reproducing the runs

With the corpus and the index built (`uv run agentic-rag ingest --download`) and Ollama serving the models on the host:

```bash
export LLM_PROVIDER=ollama EMBEDDING_PROVIDER=huggingface OLLAMA_MODEL=qwen2.5:7b-instruct
uv run agentic-rag eval                                          # the full graph
uv run agentic-rag eval --target node --node analyze_request      # routing alone
uv run agentic-rag eval --target node --node run_rag_subtask      # retrieval alone
OLLAMA_MODEL=qwen3.5:4b OLLAMA_REASONING=false uv run agentic-rag eval --judge-model qwen2.5:7b-instruct
```

Each run writes `eval-<target>-<UTC time>.json` and a Markdown summary of the same name to `data/eval/results/`. The committed runs used the `hf.co/bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M` build for `OLLAMA_MODEL` and `--judge-model`, the same weights and quantization as `qwen2.5:7b-instruct`. In fake mode (`LLM_PROVIDER=fake EMBEDDING_PROVIDER=fake`) the same commands run offline and check the pipeline, without judged metrics; their numbers say nothing about quality.
