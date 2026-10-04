# Functional evaluation

> **Status:** Phase 7, 2026-10-03; re-measured on 2026-10-04 after the [RAG reliability update](rag-improvements.md), with the same models, judge and corpus. Every number below comes from a report committed in [`data/eval/results/`](../data/eval/results/), except the first run before the fixes and the timing of one single call, and can be reproduced with the commands at the end. The load test of Phase 8 is in [performance.md](performance.md); it confirmed the conclusion on the model, and `qwen3.5:4b` with thinking off became the default.

## Contents

1. [Method](#method)
2. [Results](#results)
3. [Findings per question](#findings-per-question)
4. [Model comparison](#model-comparison)
5. [RAG reliability update](#rag-reliability-update)
6. [Holdout set](#holdout-set)
7. [How reliable the judge is](#how-reliable-the-judge-is)
8. [What the evaluation changed](#what-the-evaluation-changed)
9. [Limitations](#limitations)
10. [Conclusions](#conclusions)
11. [Reproducing the runs](#reproducing-the-runs)

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

Five questions are in Hungarian: the corpus is English, so they test the cross-lingual path (the English query rewrite) and the Hungarian answers. The comparisons q09 and q10 also list one evidence group per framework (`expected_document_groups`). A separate set of 24 questions is held out from development; see [Holdout set](#holdout-set).

### Metrics

- **Routing accuracy:** the route `analyze_request` chose against `expected_intent` (16 questions; q17 accepts any route).
- **Retrieval hit@4:** whether a retrieve sub-task ranked a chunk of an expected document among its first `TOP_K = 4` chunks, counted after the grading step, so a relevant chunk that the grading drops is a miss (10 questions).
- **Complete evidence@4:** for a question with evidence groups, whether every group has a hit among the first four chunks of a retrieve sub-task; the documents of one group are alternatives (q09 and q10).
- **Answer correctness:** a judge model compares the answer with the reference answer in substance and picks *correct* (1), *partially correct* (0.5) or *incorrect* (0) (17 questions).
- **Faithfulness:** the judge checks every claim of the answer against the material of the run, the documentation excerpts and the tool outputs: *supported* (1), *partially supported* (0.5) or *unsupported* (0). It applies to every run that searched or called a tool, also when that produced nothing, so invented figures count; direct replies are not scored (16 questions).

The judge picks one of three verdicts instead of writing a score, because a small local model gives a verdict far more consistently than a number. `architecture.md` describes the [evaluation harness](architecture.md#evaluation).

### Setup

- Windows 11, NVIDIA RTX 5070 Laptop GPU (8 GB), Ollama 0.35.0 on the host.
- LLM: Qwen2.5-7B-Instruct, 4-bit `Q4_K_M` (the `hf.co/bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M` build, the same model and quantization as the `qwen2.5:7b-instruct` tag, the default until Phase 8), temperature 0, `OLLAMA_NUM_CTX=8192`. For the comparison: Qwen3.5-4B (`qwen3.5:4b`), with and without its thinking mode.
- Judge: the same Qwen2.5-7B build in every run, also when Qwen3.5 answers (`--judge-model`).
- Embeddings `intfloat/multilingual-e5-small` on the CPU, the full index (18 654 chunks), `TOP_K=4`, `GRADE_WITH_LLM=true`, `MAX_RETRIES=2`.
- Model builds, as `ollama list` shows them: `qwen3.5:4b` is `2a654d98e6fb` and the judge `eb180556ed65`. The `qwen3.5:4b` tag has moved since: a fresh pull on 2026-10-04 gets `d8b0f5e9760c`, repackaged with a separate vision projector, which these numbers do not cover.
- The runs of 2026-10-04 use the [RAG reliability update](rag-improvements.md) with its defaults (`RETRIEVAL_CANDIDATES=20`, `HYBRID_SEARCH=true`) and otherwise the same host, Ollama server, model builds, judge and corpus snapshot; the index was updated to the revision-aware chunk ids, with the same 18 654 chunks.

## Results

Three runs of the full graph per configuration, because the runs vary (see [Limitations](#limitations)); the table gives the mean and, in brackets, the range:

| Model | Routing | hit@4 | Correctness | Faithfulness | Latency per question (median) |
|---|---|---|---|---|---|
| Qwen2.5-7B (the default until Phase 8) | 0.88 | 0.90 | 0.88 (0.82–0.91) | 0.91 (0.88–0.94) | 8.4 s |
| Qwen3.5-4B, thinking off (the default since Phase 8) | 1.00 | 0.90 | 0.91 | 0.94 | 12 s ¹ |
| Qwen3.5-4B, thinking on (one run) | 1.00 | 0.90 | 0.91 | 0.94 | 143 s |
| Qwen3.5-4B, thinking off, RAG reliability update (2026-10-04) | 1.00 | 1.00 | 0.91 | 1.00 | 13 s ¹ |

¹ Inflated: the two models do not fit in the 8 GB of VRAM together, so Ollama reloads Qwen3.5 after every judge call. The node runs below have no judge.

Complete evidence@4 is 1.00 in every graph run, before the update too (recomputed from the rankings stored in the reports): the planner gives each framework of a comparison its own search.

The node targets measure one step alone, without a judge (one run each):

| Model | Routing alone (`analyze_request`) | Median call | Retrieval alone (`run_rag_subtask`), hit@4 | Median call |
|---|---|---|---|---|
| Qwen2.5-7B | 0.88 | 0.9 s | 0.80 | 0.5 s |
| Qwen3.5-4B, thinking off | 1.00 | 0.8 s | 0.90 | 0.6 s |
| Qwen3.5-4B, RAG reliability update | 1.00 | 0.8 s | 1.00 | 1.5 s |
| Qwen3.5-4B, update with `HYBRID_SEARCH=false RETRIEVAL_CANDIDATES=4` | – | – | 0.90 | 0.6 s |

Retrieval alone runs one search for the whole question, so a comparison has complete evidence only if that one search finds both frameworks: q09 does, q10 (React and Nuxt `useState`) does not, in both configurations (0.50).

By language and kind of question, mean of the three runs per configuration:

| Group | Qwen2.5-7B correctness | Qwen2.5-7B faithfulness | Qwen3.5-4B correctness | Qwen3.5-4B faithfulness | Update correctness | Update faithfulness |
|---|---|---|---|---|---|---|
| English (12) | 0.92 | 0.97 | 0.92 | 0.91 | 0.92 | 1.00 |
| Hungarian (5) | 0.80 | 0.77 | 0.90 | 1.00 | 0.90 | 1.00 |
| Single search (8) | 0.85 | 0.98 | 0.81 | 0.88 | 0.88 | 1.00 |
| Multi-part (3) | 0.78 | 0.61 | 1.00 | 1.00 | 1.00 | 1.00 |
| Tool (5) | 0.90 | 0.80 | 1.00 | 1.00 | 0.90 | 1.00 |

The question-by-question tables are in the Markdown summaries next to every report, for example [`eval-graph-20261003T141916Z.md`](../data/eval/results/eval-graph-20261003T141916Z.md).

## Findings per question

With Qwen2.5-7B, the default at the time:

- **Retrieval works for both languages.** hit@4 is 0.90 on the graph: the English query rewrite brings the Hungarian questions (q07, q08, q10) to the right pages. The only miss is q03: for *computed property with the Composition API* the search returns the Options API reference (`vue/api/options-state.md`) or the Composition API FAQ, never the guide page `computed.md`. Retrieval alone (one search for the whole question) scores 0.80: the graph also recovers q02 through its plan.
- **Routing errs on two tool questions, every time.** The specificity question (q12) goes to a search, and the two-contrast question (q15) to a single tool call. The verification and the re-plan repair q12 in every run (the answer is correct); q15 is answered correctly in two runs out of three, and in the third, one ratio is invented (*4.5:1* for black on `#f5f5f5`, really 19.26:1).
- **The weak spot is Hungarian.** The Hungarian answers score 0.80 for correctness and 0.77 for faithfulness, against 0.92 and 0.97 in English. The multi-part Hungarian question (q10) mixes the two frameworks' terms (*both are Hooks*) in two runs of three, and q15 is the worst question of the set.
- **Out of scope and tools are reliable.** q17 is declined in every run, q11, q13 and q14 give the tool's verdict, q16 greets without a search.

## Model comparison

Qwen3.5-4B with its thinking mode switched off (`OLLAMA_REASONING=false`) answers better than the 7B model on every metric that differs: it routes all 16 questions right (q12 and q15 included, in all three node and graph runs), retrieves better alone (0.90 against 0.80, through a better English rewrite), and its Hungarian and multi-part answers are correct and faithful in every run. It gave the same scores in all three runs, while the 7B model varied. Its one systematic failure is q03: it follows the retrieved Options API reference faithfully and gets the Composition API wrong (correctness 0), where the 7B model answers partly from general knowledge. The RAG reliability update fixes q03 through the retrieval (see below).

With thinking on, the quality is the same, but every call writes a long hidden reasoning first: a question took 143 s at the median, against 12 s with thinking off. One call measured on its own: 13.8 s with thinking, 0.7 s without. Thinking is therefore useless for this agent, which makes four to eight LLM calls per question.

The model is 3.4 GB instead of 4.7 GB, so it leaves more of an 8 GB GPU for the KV cache of parallel requests. Per call it is as fast as the 7B model (0.8 s against 0.9 s routing, 0.6 s against 0.5 s retrieval); the end-to-end latency without the judge's model swaps is what the load test measured: 3.6 s at the median for one user, and 28 % more requests per minute than the 7B model with four ([performance.md](performance.md#consequence-for-the-model-choice)).

## RAG reliability update

The [update](rag-improvements.md) changes the retrieval (a BM25 keyword ranking fused with the vector ranking, 20 candidates graded, up to four kept), the verification (an unreadable verdict withholds the draft) and the tool path (an exact single tool result skips the answer and the verification calls). Measured with the same models, judge and corpus as the Qwen3.5 rows above, three runs with identical scores (reports `eval-graph-20261004T191858Z`, `…192448Z`, `…193039Z`):

- **q03 is fixed.** The guide page `vue/guide/essentials/computed.md` now ranks in the top four, and the answer is correct and faithful (both 0 before). The same code with the vector-only configuration (`HYBRID_SEARCH=false RETRIEVAL_CANDIDATES=4`) still misses the page, so the keyword ranking and the larger pool make the difference. This was the one unfaithful answer, so faithfulness reaches 1.00.
- **Two correct answers lose half a point to the judge.** q05 adds that the state of `useState` is reachable as `payload.state.[name-of-your-state]`; the correctness judge calls it inaccurate, but the sentence is verbatim in `nuxt/4.api/2.composables/use-nuxt-app.md`, and the faithfulness judge rates it supported. q13 is now the exact `browser_support` output, which ends in *Safari 15: not supported (added in 15.4).*; the judge reads it as contradicting the reference, which says the same. Without these two errors the correctness would be 0.97.
- **Tool answers are faster, searches slightly slower.** The tool questions q11–q14 take 6.8–7.1 s instead of 8.9–10.2 s (median per question, judge swaps included), because the answer and its verification are no longer generated; the single searches take up to 3 s longer, mostly in the grading call over 20 candidates. The load test measures both without the judge ([performance.md](performance.md#rag-reliability-update)).
- **Routing and complete evidence stay at 1.00**, alone and in the graph.

## Holdout set

[`data/eval/holdout.jsonl`](../data/eval/holdout.jsonl) holds 24 questions that no prompt or retrieval change was tuned on: 18 in English and 6 in Hungarian, two follow-ups with a fixed conversation history, three comparisons with evidence groups, a question that is ambiguous between frameworks, three version-boundary questions (Nuxt 2, the Next.js Pages Router, Safari 15.3), an API that does not exist, two out-of-scope questions, four tool questions and a mixed tool and search request. Its reference answers are drafts written with the update and not yet reviewed by a domain expert, so its numbers are indicative. One run with the setup above (report `eval-graph-20261004T194127Z`):

| Routing | hit@4 | Complete evidence@4 | Correctness | Faithfulness | Latency per question (median) |
|---|---|---|---|---|---|
| 0.94 (18) | 0.81 (16) | 1.00 (3) | 0.67 (24) | 0.73 (24) | 14 s ¹ |

A manual review of the 14 answers scored below 1 on either judged metric sorts the failures:

- **Judge errors (3).** h18 and h20 are exact tool outputs (*Contrast ratio 1.00:1 … fails*; *Safari 15.3: not supported (added in 15.4).*) that state the reference answer, and h23 correctly says that the indexed React documentation has no `useQuantumState` hook; all three are scored 0. Without them the correctness is 0.79.
- **Version boundaries (h15, h16).** The corpus covers the current Nuxt and the App Router only. Asked whether `useState` works unchanged in Nuxt 2, the answer attributes a Nuxt 4 → 5 change of `clearNuxtState` to Nuxt 2; asked about the Pages Router, it gets the main point right (*no*) but adds wrong details. Both are unfaithful (0).
- **Ambiguity (h14).** *Which useState API should I use? I have not chosen a framework yet.* gets a Nuxt-only answer instead of a question back or a comparison: the search returned Nuxt pages only, and nothing in the workflow asks the user to clarify.
- **Model errors (h01, h09, h13, h24).** h01 explains React's batching wrongly (*the second call sets `(count + 1) + 1`*), although it retrieved the right guide page, *Queueing a Series of State Updates*, which the item's expected documents do not list; h09 suggests updating Nuxt state through `nuxtApp.payload.state`; the Hungarian follow-up h13 opens with *Igen* (yes) and then explains correctly that templates need no `.value`; h24 declares a second `useState` in a child component, which would not change the shared theme.
- **Answer language.** The English question h02 is answered in Hungarian, and the Hungarian tool question h18 gets the English tool output verbatim, as the fast path is designed to.

The routing errs once (h02, planned as a comparison), the retrieval misses three expected pages (h01, h02 and h16), and both follow-ups find the right page through their history.

## How reliable the judge is

A manual review of the 17 answers of the first committed run agrees with 15 of the judge's correctness verdicts:

- **q02** (*correct*, judged partially correct): the answer shows a class component before `useState`; it is correct, and the judge's rationale invents a contradiction about React versions.
- **q08** (*correct*, judged partially correct): the judge reads the documented order of cleanup and setup as an error.

The faithfulness verdicts are mostly right but lenient once: in q03 the answer's `computed` example is not in the retrieved material, and the judge still rates it supported. The judge did catch the invented ratio of q15 and the wrong API of q03 (Qwen3.5). On these numbers the judge is slightly too strict on correctness, so the correctness scores are, if anything, a little low; the comparison between the models is not affected, because the same judge scores both.

After the update and on the holdout it errs the same way: it under-scores short exact answers, the verbatim `browser_support` and `check_contrast` outputs of q13, h18 and h20 and the correct refusal of h23, and it calls a documented detail inaccurate (q05). Correcting these five verdicts gives 0.97 on the development set and 0.79 on the holdout. Giving the correctness judge the tool outputs and excerpts, as the faithfulness judge already gets them, would likely remove most of them.

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
| Qwen3.5-4B took minutes per question | `OLLAMA_REASONING` turns the thinking mode of reasoning models off (`false`) or on; unset kept the model's default (since Phase 8 the default is `false`) |

The router examples are not questions of the set, and the expected documents were only extended with pages that contain the answer, so the set still measures generalization rather than the prompt.

## Limitations

- **Seventeen questions.** One question moves a metric by about 0.06 (by 0.1 to 0.2 in a group of five), so differences below 0.1 are noise.
- **Run-to-run variance.** Even at temperature 0 the 7B runs differ (correctness 0.82 to 0.91): the parallel searches of a complex question reach Ollama at the same time, and a different first draft changes the verifier's decision and the re-plan. Qwen3.5 gave identical scores in all three runs.
- **Self-judging.** The 7B model judges its own answers; only local models are allowed, and none is clearly stronger. The [manual review](#how-reliable-the-judge-is) shows where the judge errs, and the model comparison keeps the judge fixed.
- **Latency.** Measured one question at a time on one machine; the first question includes loading the models (about 15 s), and the Qwen3.5 graph runs include the judge's model swaps. The load test measures latency properly.
- **Few follow-ups.** Every question of the development set runs on a fresh state; the holdout adds two follow-ups with a fixed history. One run of 24 questions with draft reference answers is indicative, not a benchmark.

## Conclusions

1. **The agentic workflow does what it is built for.** With the 7B model, the default at the time, the answers score 0.88 for correctness and 0.91 for faithfulness on average; retrieval finds the right page for 9 of 10 questions in both languages, the out-of-scope question is declined, and the tools deliver exact verdicts that are always shown verbatim.
2. **The weak points are routing tool questions and Hungarian.** The 7B router misroutes two of the five tool questions; the verification and the re-plan repair q12 in every run and q15 in two runs of three. Hungarian answers lose about 0.12 in correctness and 0.20 in faithfulness against English.
3. **Decision 4 changes: Qwen3.5-4B with thinking off is the better model here.** It fixes both weak points (routing 1.00, Hungarian 0.90 and 1.00), is stable from run to run and smaller. It must run with `OLLAMA_REASONING=false`: with thinking on it is about ten times slower without being better. The load test (Phase 8) compared the two models' latency under load before the default changed; Qwen3.5-4B was faster there too, and it is the default since ([performance.md](performance.md#consequence-for-the-model-choice)).
4. **The hybrid search closes the retrieval gap.** Before the update, the Vue guide page on computed properties was outranked by the API reference (hit@4 0.90). The keyword ranking of the RAG reliability update puts it into the top four (hit@4 1.00, alone and in the graph), and faithfulness reaches 1.00; correctness stays at 0.91 only because the judge misreads two correct answers.
5. **The holdout shows the next work.** Questions across version boundaries, ambiguous questions and the answer language fail where the development set has no examples (correctness 0.67, 0.79 without the judge's errors). They need a version-aware refusal, a clarifying question and a fixed answer language, and the judge needs the evidence to score short exact answers.

## Reproducing the runs

With the corpus and the index built (`uv run agentic-rag ingest --download`) and Ollama serving the models on the host:

```bash
export LLM_PROVIDER=ollama EMBEDDING_PROVIDER=huggingface OLLAMA_MODEL=qwen2.5:7b-instruct
uv run agentic-rag eval                                          # the full graph
uv run agentic-rag eval --target node --node analyze_request      # routing alone
uv run agentic-rag eval --target node --node run_rag_subtask      # retrieval alone
OLLAMA_MODEL=qwen3.5:4b OLLAMA_REASONING=false uv run agentic-rag eval --judge-model qwen2.5:7b-instruct
```

The runs of 2026-10-04, with the RAG reliability update and its defaults (`qwen3.5:4b` and thinking off are the defaults too):

```bash
uv run agentic-rag ingest                                         # once: revision-aware chunk ids
uv run agentic-rag eval --judge-model qwen2.5:7b-instruct         # three times
uv run agentic-rag eval --target node --node analyze_request
uv run agentic-rag eval --target node --node run_rag_subtask
HYBRID_SEARCH=false RETRIEVAL_CANDIDATES=4 uv run agentic-rag eval --target node --node run_rag_subtask
uv run agentic-rag eval --dataset data/eval/holdout.jsonl --judge-model qwen2.5:7b-instruct
```

Each run writes `eval-<target>-<UTC time>.json` and a Markdown summary of the same name to `data/eval/results/`. The committed runs used the `hf.co/bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M` build for `OLLAMA_MODEL` and `--judge-model`, the same weights and quantization as `qwen2.5:7b-instruct`. In fake mode (`LLM_PROVIDER=fake EMBEDDING_PROVIDER=fake`) the same commands run offline and check the pipeline, without judged metrics; their numbers say nothing about quality.
