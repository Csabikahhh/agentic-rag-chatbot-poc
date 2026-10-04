# Agentic RAG Chatbot – Proof of Concept

**English** | [Magyar](README.hu.md)

An agentic Retrieval-Augmented Generation (RAG) chatbot prototype in Python — built with [LangGraph](https://github.com/langchain-ai/langgraph), powered by a locally hosted open-source LLM, with a [Streamlit](https://streamlit.io/) UI, and fully containerized with Docker.

> **Status:** complete (2026-10-03): every phase of the [build order](docs/project-structure-plan.md#8-build-order) is done. A [frontend developer assistant](#problem-statement-and-motivation) over the official MDN, React, Vue, Next.js, Nuxt and TypeScript documentation, with three non-retrieval tools (decisions 8–9 of the [project structure plan](docs/project-structure-plan.md)): `agentic-rag ingest --download` downloads the documentation at pinned commits, cleans and chunks it and builds the vector index; the RAG subgraph rewrites a question into an English search query, retrieves and grades the chunks and returns a cited context; the main workflow routes every question, splits complex ones into parallel searches and tool calls (contrast, specificity, browser support), writes a cited answer and verifies it; the Streamlit UI shows every step live, the RAG subgraph's steps under each search, and the sources of the answer; `docker compose up --build` runs the whole stack from a fresh clone, downloading the model and the corpus and building the index on its own. A functional evaluation of 17 questions measures routing, retrieval, correctness and faithfulness, and a load test finds the bottleneck (see [Evaluation](#evaluation)). CI runs the lint, the offline tests and the image build on every push.

> **RAG reliability update (2026-10-04):** hybrid BM25/vector retrieval now considers 20 candidates and keeps up to four relevant chunks. Malformed verification withholds the draft; exact single-tool answers skip two model calls. Citations link to their sources and show indexed snapshot metadata after re-ingestion. A separate 24-question holdout and a complete-evidence metric extend the evaluation. Re-measured with the same models as the 3 October baseline: retrieval hit@4 0.90 → 1.00, faithfulness 0.94 → 1.00, correctness 0.91 as before (the two half-points it lost are judge errors), the same throughput; the holdout shows where the system still fails (see [Evaluation](#evaluation)). Details and migration: [docs/rag-improvements.md](docs/rag-improvements.md).

## Contents

1. [Overview](#overview)
2. [Assignment requirements](#assignment-requirements)
3. [Problem statement and motivation](#problem-statement-and-motivation)
4. [System architecture](#system-architecture)
5. [Design decisions](#design-decisions)
6. [Evaluation](#evaluation)
7. [Getting started](#getting-started)
8. [Repository structure](#repository-structure)
9. [License](#license)

## Overview

The goal is a working, well-documented and reproducible prototype that demonstrates:

- **Agentic workflow design with LangGraph** — autonomous decisions through conditional routing, splitting complex requests into sub-tasks that are executed independently, and explicit state management for intermediate results.
- **A modular RAG subsystem** — a dedicated LangGraph subgraph, invoked from the main workflow.
- **Tool use beyond retrieval** — at least one tool that does more than search the knowledge base.
- **Fully local inference** — an open-source LLM, no paid APIs.
- **Measured quality and performance** — a mini evaluation set and a load test with bottleneck analysis.

**Tech stack:** Python, LangGraph, Streamlit, Docker / Docker Compose and a local open-source LLM (see [Design decisions](#design-decisions)).

This project is a solution to a technical assignment for a Medior AI Engineer position; the original brief (in Hungarian) is in the local `task/` folder, which is gitignored.

## Assignment requirements

Status of each requirement from the brief:

**Problem & data**

- [x] Real-world problem (domain / use case) with a written justification
- [x] Freely chosen text data source, with the focus on quality processing and scalable data integration rather than volume

**Agentic architecture (LangGraph)**

- [x] Agentic workflow with at least 5 nodes
- [x] Autonomous decision-making (e.g. conditional routing)
- [x] Decomposition into sub-tasks and their independent execution
- [x] State management for storing intermediate results
- [x] At least 2 tools, at least one of which is not purely retrieval-based
- [x] Dedicated, modular RAG subgraph callable from the main workflow (not counted towards the node count)

**Model, UI & deployment**

- [x] Open-source LLM that fits the local resources (no paid APIs), with the trade-offs justified
- [x] Streamlit prototype UI that shows the agent's main steps and the result of the RAG process
- [x] Containerized: `Dockerfile` (required) and `docker-compose.yml` (a plus for multi-component setups)

**Evaluation & performance**

- [x] Functional evaluation on a mini set of 10–20 questions (a single node or the full workflow)
- [x] Load test with 50–200 queries: basic latency metrics, the main bottleneck, 1–2 concrete optimization proposals

**Documentation**

- [x] This README: problem & goals, architecture & design rationale, evaluation & load-test results, setup & run guide

## Problem statement and motivation

**Use case: a frontend developer assistant.** The chatbot answers questions about web frontend development from the official documentation: HTML, CSS, JavaScript and TypeScript, accessibility, and the React, Vue, Next.js and Nuxt frameworks. Concrete facts are checked with deterministic tools: browser support, colour contrast and CSS specificity. An operations extension (the Kubernetes and Docker documentation, with manifest and schedule checks) follows once the frontend scope works; it adds sources and tools, not a new architecture.

- **Why is the problem relevant?** Frontend knowledge is spread over many sources that change quickly: the web platform reference (MDN), the documentation of each framework and its versions, and the accessibility guidelines. General-purpose LLMs answer such questions fluently but unreliably. They invent APIs, mix up framework versions (the Pages and the App Router of Next.js, Nuxt 2 and the current Nuxt) and APIs of the same name (React's `useState` hook and Nuxt's `useState` composable), and they cannot tell whether a feature works in the browsers a project has to support.
- **What user need does it address?** Developers need short, correct answers with a link to the source, and exact results for the questions that have one: *does `:has()` work in Safari 15?*, *does this grey text pass WCAG AA?*, *why does this rule not apply?* The assistant answers from pinned versions of the official documentation, cites every claim, and says so when the documentation does not cover a question.
- **Why is an agentic RAG approach a good fit?** Real questions mix explanation with verification and often span several frameworks, so a single retrieve-then-generate pass falls short:
  - a request such as *How do I fetch data on the server in Next.js and in Nuxt, and what is the difference?* is split into one retrieval per framework, run in parallel and combined into one comparison;
  - facts that can be computed are computed by a tool, not guessed by the model: `#777777` text on white has a contrast ratio of 4.47:1, just below the 4.5:1 that AA requires, a margin a model easily gets wrong;
  - the verification step checks the draft against the retrieved documentation and re-plans when the draft is not supported, which catches invented APIs before they reach the user.

## System architecture

High-level target architecture:

```mermaid
flowchart LR
    user([User]) <--> ui[Streamlit UI]
    subgraph app [LangGraph agent]
        main["Main agentic workflow<br/>≥ 5 nodes · conditional routing · shared state"]
        rag[["RAG subgraph"]]
        main <--> rag
    end
    ui <--> main
    main <--> tools["Tools<br/>≥ 1 non-retrieval tool"]
    main <--> llm["Local open-source LLM"]
    rag <--> index[("Document index")]
    source["Text data source"] -. ingestion .-> index
```

| Component | Role |
|---|---|
| Streamlit UI | Chat interface that visualizes the agent's steps and the retrieved sources |
| Main agentic workflow | Plans, routes and executes sub-tasks; keeps intermediate results in the graph state |
| RAG subgraph | Modular retrieval pipeline over the document index, called from the main workflow |
| Tools | Capabilities beyond retrieval — at least one non-retrieval tool |
| Local LLM | Open-source model served locally, no paid APIs |
| Document index | Chunked and embedded documents from the chosen text data source |

The parts that are built:

- **Ingestion pipeline:** `data/sources.toml` → sparse git download at pinned commits → dialect cleaning and one document per H2/H3 section → structure-aware chunks with a context line → Chroma, where only new or changed chunks are embedded.
- **RAG subgraph** (`rewrite_query` → `retrieve` → `grade_documents` → `build_context`, linear): the chat model rewrites the question into one English search query; Chroma and a local BM25 keyword index rank the chunks, and reciprocal-rank fusion keeps `RETRIEVAL_CANDIDATES` (20) candidates; a score threshold for the vector-only hits and one LLM grading call drop the irrelevant ones, and up to `TOP_K` (4) of the rest become a context with the citation markers `[1]`, `[2]`, … and the matching sources. Every step records a trace event with its duration and a one-line summary.

- **Main workflow** (seven nodes): `analyze_request` classifies the message (`direct`, `single`, `tool`, `complex`); a single question goes straight to one search, a tool question to one tool call, and a complex one to `plan_subtasks`, which plans up to five independent sub-tasks that LangGraph runs in parallel (`Send`); `synthesize_answer` writes a cited answer from the results (after a single successful deterministic tool call it makes no model call and the verification is skipped: the exact tool output is the answer), `verify_answer` checks it and re-plans what is missing at most `MAX_RETRIES` times (a verification that cannot be read withholds the draft), and `finalize_response` returns the answer, its numbered sources and the verbatim tool outputs.
- **Tools:** `search_knowledge_base` (the RAG subgraph), `check_contrast` (WCAG 2.2 contrast), `css_specificity` (Selectors Level 4) and `browser_support` (MDN browser-compat-data, pinned).

The detailed design (the seven main nodes and their routing, the four steps of the RAG subgraph with their prompts and thresholds, the ingestion pipeline, the tools), the state contracts as they exist in the code, the cross-cutting contracts (dependency injection, the execution model, trace events, the re-plan loop, citation numbering, errors and exit codes) and the configuration reference are in [docs/architecture.md](docs/architecture.md).

> Tip: `uv run agentic-rag export-graph` prints both compiled graphs as Mermaid (through `graph.get_graph(xray=True).draw_mermaid()`); the diagrams in [docs/architecture.md](docs/architecture.md) are generated this way. The RAG subgraph gets a diagram of its own: the main workflow calls it inside the `run_rag_subtask` node, through the `search_knowledge_base` tool, where `xray=True` does not expand it.

## Design decisions

Decisions 1–7 of the [project structure plan](docs/project-structure-plan.md#3-decisions-to-lock-in-before-scaffolding) are built into the code. Decisions 8–9, the domain and the non-retrieval tools, were made on 2026-10-02: the corpus and its ingestion (Phase 2) and the tools (Phase 4) are built. The LLM and the embedding model (decisions 4 and 5) and the chunking parameters are provisional until the evaluation and the load test have measured them.

| Area | Key trade-offs | Choice & rationale |
|---|---|---|
| Domain & data source | Relevance, availability and licensing, preprocessing effort | **Frontend developer assistant** over the official documentation (Phase 2): MDN Web Docs (a curated subset on CSS, HTML, accessibility and JavaScript; prose CC BY-SA 2.5, code samples CC0), React (CC BY 4.0), Vue (CC BY 4.0), Next.js (MIT), Nuxt (MIT) and the TypeScript Handbook (CC BY 4.0). `agentic-rag ingest --download` fetches them from pinned commits, as listed in `data/sources.toml`: every run indexes the same versions, and no share-alike text enters the repository. The documentation is versioned, structured and covers the questions developers actually ask. The operations extension (Kubernetes, CC BY 4.0; Docker, Apache 2.0) is two more entries in the same list |
| Non-retrieval tools | Fit to the domain; deterministic, local and testable | **Three tools** (Phase 4), `browser_support`, `check_contrast` and `css_specificity`: *browser support* looks a feature up in MDN's `browser-compat-data` (CC0, downloaded at a pinned commit like the corpus, as a source with `index = false`), resolves BCD's `mirror` statements and compares the versions with the target browsers; *colour contrast* computes the WCAG 2.x contrast ratio of two colours and whether it passes AA and AAA for normal and large text; *CSS specificity* computes the specificity of selectors by the Selectors Level 4 rules and tells which one wins. Each is a computation or a lookup in pinned data, so it can be tested exactly, and it gives the model facts it would otherwise guess |
| Packaging & Python version | Reproducible builds, wheel coverage of the ML stack, setup effort | **uv** (`pyproject.toml` + `uv.lock`), **Python 3.12**, `src/` layout: the lock file pins every package for local runs and the image alike, uv installs the pinned Python itself, and 3.12 has the widest wheel coverage for torch and chromadb |
| LLM | Answer quality vs. latency vs. memory (RAM/VRAM); tool-calling support; license | **`qwen3.5:4b` with its thinking mode off** (`OLLAMA_REASONING=false`), chosen by the evaluation and the load test: a multilingual model under the Apache 2.0 license, 3.4 GB at 4 bits, which leaves room in the 8 GB of VRAM of the development machine. Against the provisional `qwen2.5:7b-instruct` it routes every evaluation question right (1.00 against 0.88), answers Hungarian questions better (correctness 0.90 against 0.80) and gives the same scores in every run; with four concurrent users it serves 28 % more requests per minute with a 39 % lower p95. Its thinking mode stays off: it made every call about ten times slower without better answers. Trade-off: Ollama does not batch its hybrid architecture, so with parallel slots the 7B transformer scales better to several users ([docs/performance.md](docs/performance.md)) |
| LLM serving | Setup effort, containerization, throughput | **Ollama** (a Compose service, or Ollama on the host) plus a **scripted fake provider**: Ollama gives an HTTP API and GPU support without compiling anything into the image; the fake (`LLM_PROVIDER=fake`) is the brief's dummy LLM and keeps the tests model-free |
| Tool-calling style | Reliability with small local models vs. flexibility of native tool calling | **Structured-output planner + explicit tool nodes**: the planner emits typed sub-tasks as JSON, which small local models produce more reliably than native tool calls; the tools stay LangChain tools, so `bind_tools` remains possible |
| Embedding model | Retrieval quality vs. speed; language coverage | **`intfloat/multilingual-e5-small`**, provisional, run locally with sentence-transformers: multilingual (Hungarian included) and small (384 dimensions), so it runs on the CPU and leaves the GPU to the LLM |
| Retrieval and grading | Recall vs. precision; latency of extra LLM calls; Hungarian questions over an English corpus | **Rewrite, hybrid search, threshold, LLM grade** (Phase 3; hybrid since the RAG reliability update): the chat model rewrites every question into one English search query (the multilingual embeddings alone missed Hungarian questions, see *Cross-lingual retrieval* below); Chroma and a BM25 keyword index over the same chunks (SQLite FTS5, built in memory on the first search, no extra model) each rank the chunks, and reciprocal-rank fusion keeps 20 candidates (`RETRIEVAL_CANDIDATES`); a score threshold per embedding provider (0.83 for E5, measured on the index) drops vector-only hits that are clearly unrelated; one structured-output call grades the remaining chunks together, so the cost is one LLM call per retrieval, not one per chunk (`GRADE_WITH_LLM=false` turns it off); up to `TOP_K` (4) relevant chunks are kept. The keyword ranking finds the one page the vectors missed (hit@4 0.90 → 1.00); the longer candidate list makes the grading call slower. Measured with Qwen3.5-4B, one request at a time: rewrite 0.31 s (median), grade 0.26 s with 4 candidates and 0.48 s with 20, search 20 ms. `HYBRID_SEARCH=false RETRIEVAL_CANDIDATES=4` restores the vector-only search |
| Vector store | Persistence, metadata filtering, scalability | **Chroma** with a persistent client in `data/chroma_db/` (a named volume in Compose): persistence and metadata filtering without pickle deserialization |
| Chunking | Chunk size and overlap vs. retrieval precision and context length | **Structure-aware:** the loaders split every page at its H2 and H3 headings, and the heading path becomes the `section` metadata. Each section is packed into chunks of up to 900 characters at paragraph boundaries; a code block stays whole up to 1 800 characters, a heading never ends a chunk, and short trailing paragraphs (up to 150 characters) are repeated in the next chunk. Every chunk starts with a context line of title and headings (`useState – React > Reference > useState(initialState)`), so a chunk such as *Parameters* still names its subject for the embedding model and the prompt. Result: 18 654 chunks from 1 160 pages, median 602 characters. The evaluation (Phase 7) measured a retrieval hit@4 of 0.90 with these sizes; its one miss is a ranking problem, not a chunking one |

Notes on the defaults:

- **Models.** The LLM was chosen with the evaluation and the load test (decision 4, above). The embedding model stays the provisional default of decision 5: with it alone the evaluation found the right page for 9 of 10 questions in both languages, and its one miss, a ranking problem, is fixed by the hybrid keyword search of the RAG reliability update (10 of 10). The documentation is in English and the questions may be Hungarian or English, so the evaluation set also checks Hungarian questions over the English corpus.
- **Model builds.** The evaluation and the load test ran with the `qwen3.5:4b` build `2a654d98e6fb` (pulled on 2026-08-27) and the judge `hf.co/bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M` (`eb180556ed65`). Ollama tags are mutable: on 2026-10-04 a fresh `ollama pull qwen3.5:4b`, as in the Compose stack, got the build `d8b0f5e9760c`, repackaged with a separate vision projector; it answered the smoke-test question correctly but is not the measured build. `ollama list` shows which build a server has.
- **Corpus scope.** The framework documentation mixes versions and legacy sections. The source list keeps the current guides and API references, such as the App Router of Next.js, and leaves out the Pages Router, Nuxt Bridge, the migration guides and MDN's vendor-prefixed selectors: 1 160 pages (MDN 573, Nuxt 172, Next.js 165, React 151, Vue 80, TypeScript 19). Every source is stored in a directory of its own under `data/raw/` (`mdn/`, `react/`, `vue/`, `nextjs/`, `nuxt/`, `typescript/`), and its name is added to every page title (`useState – React`, `useState – Nuxt`), so every citation shows which documentation it comes from, and APIs of the same name stay apart.
- **Cleaning.** Each documentation set writes Markdown in its own dialect. The loaders turn MDN's macros, the JSX components of React and Next.js, Vue's VitePress containers and Nuxt's MDC components into plain Markdown, drop the Pages Router blocks of the Next.js pages, replace links by their text, and keep every code block verbatim (details in `src/agentic_rag/ingestion/markdown.py`).
- **Cross-lingual retrieval.** A first check on the built index confirms the risk of a Hungarian question over an English corpus. English questions find the right page: *Which CSS pseudo-class selects a parent element that contains a specific child?* returns MDN's `:has()` first, and *How do I add state to a React component?* returns the state sections of `Component` and `useState`. The Hungarian question *Hogyan kérek le adatot szerveroldalon Next.js App Routerben?* ("How do I fetch data on the server in the Next.js App Router?") does not reach the *Fetching Data* page in the top three. The `rewrite_query` step of the RAG subgraph (Phase 3) therefore turns every question into an English search query before retrieval. With it, the question becomes *How do I fetch data on the server in Next.js App Router?* and retrieves the *Fetching Data* page; *How do I create a dynamic route in the Next.js App Router?*, which first found a React page, becomes *next.js app router dynamic route* and finds *Dynamic Route Segments*. The evaluation (Phase 7) confirms it on more questions: the Hungarian questions retrieve the right pages as often as the English ones.
- **Embeddings.** The no-paid-API rule rules out hosted embedding APIs, so the embeddings run locally. The default model is downloaded once (about 0.5 GB, into the Hugging Face cache, `HF_HOME`) and then works offline; it takes a few seconds to load, its E5 `query:` / `passage:` prefixes are added automatically, and it brings CPU-only torch into the image (about 0.8 GB of the 2.9 GB image). `EMBEDDING_PROVIDER=fake` replaces it with deterministic, hashed bag-of-words vectors: offline and instant, but purely lexical, so only for tests and model-free demos.
- **Rebuilding the index.** Vectors of different models are not comparable: after changing `EMBEDDING_PROVIDER` or `EMBEDDING_MODEL`, rebuild the index with `agentic-rag ingest --rebuild`.

## Evaluation

### Functional evaluation

**Set:** 17 questions over the documentation in [data/eval/questions.jsonl](data/eval/questions.jsonl): 8 single searches across the six sources, 3 multi-part questions, 4 tool questions, a greeting and a question outside the topic; 5 of them in Hungarian. Each has a reference answer, the documents that answer it and the expected route ([data/eval/README.md](data/eval/README.md)). A separate **holdout** of 24 questions, [data/eval/holdout.jsonl](data/eval/holdout.jsonl), is kept out of development: follow-ups with a fixed conversation history, Hungarian questions, framework ambiguity, version boundaries, an API that does not exist and mixed tool and search requests. Its reference answers are drafts that a domain expert has not reviewed yet.

**Metrics:** routing accuracy, retrieval hit@4 (per retrieve sub-task, after grading), complete evidence@4 (a comparison needs a hit for every framework it asks about), and answer correctness and faithfulness judged by a local LLM, which picks one of three verdicts (1, 0.5 or 0). `agentic-rag eval` runs the full graph, or one node: `--node analyze_request` for routing alone, `--node run_rag_subtask` for retrieval alone. Every run writes a JSON report and a Markdown summary to `data/eval/results/`.

**Results** (mean of three runs each, RTX 5070 Laptop GPU, Qwen2.5-7B as the judge in every run):

| Model | Routing | hit@4 | Correctness | Faithfulness | Hungarian correctness | Median latency |
|---|---|---|---|---|---|---|
| `qwen2.5:7b-instruct` (the former default) | 0.88 | 0.90 | 0.88 | 0.91 | 0.80 | 8.4 s |
| `qwen3.5:4b`, thinking off (default since Phase 8), 3 October | 1.00 | 0.90 | 0.91 | 0.94 | 0.90 | 12 s (inflated by the judge's model swaps; 3.6 s in the load test) |
| `qwen3.5:4b`, thinking on (one run) | 1.00 | 0.90 | 0.91 | 0.94 | 0.90 | 143 s |
| `qwen3.5:4b`, thinking off, **RAG reliability update**, 4 October | 1.00 | **1.00** | 0.91 | **1.00** | 0.90 | 13 s (inflated as above; 3.9 s in the load test) |

Complete evidence@4 is 1.00 in every run, before the update too (recomputed from the stored rankings): the planner gives each framework of a comparison its own search. The three runs of a Qwen3.5 configuration give identical scores.

**Holdout** (one run, the same models and judge): routing 0.94, hit@4 0.81, complete evidence@4 1.00, correctness 0.67, faithfulness 0.73. A manual review of the answers scored below 1 finds three judge errors (two exact tool outputs and a correct refusal, all scored 0; correctness 0.79 without them) and real failures that the development set does not provoke: questions across a version boundary (Nuxt 2, the Next.js Pages Router) answered with unsupported details, a framework-ambiguous question answered for one framework instead of asking, a wrong explanation of how React batches state updates, and an English question answered in Hungarian.

**Conclusions:**

- The workflow does what it is built for: retrieval finds the right page for every question of the development set in both languages, the out-of-scope question is declined, and the tools give exact verdicts, shown verbatim in every answer.
- With the former default, the 7B model, the weak points were the routing of tool questions (two of five go to a search or to a single tool call; the verifier repairs most of these) and Hungarian answers (0.12 lower correctness, 0.20 lower faithfulness than English).
- `qwen3.5:4b` with thinking off fixes both and is stable from run to run; after the load test confirmed it, it became the default (decision 4).
- The RAG reliability update keeps every score and improves two: the keyword ranking finds the page the vectors missed (hit@4 1.00; with `HYBRID_SEARCH=false RETRIEVAL_CANDIDATES=4` the same code still misses it), and every answer is faithful (1.00). Correctness stays at 0.91 because the judge gave half a point to two correct answers: one adds a detail that is verbatim in the Nuxt documentation, the other is the exact `browser_support` output.
- The holdout is harder than the development set and shows the next work: answers across version boundaries, a clarifying question for ambiguous requests, and the answer language; the judge also needs care with short exact answers.
- The evaluation found and fixed defects first: an out-of-scope question answered from general knowledge, a two-contrast question that invented its ratios after a failed tool call, mislabelled tool outputs, and a judge that rewarded both.

[docs/evaluation.md](docs/evaluation.md) has the details: the question-by-question findings, a manual review of the judge (it agrees with 15 of 17 verdicts of the first run and errs on the strict side; its errors after the update and on the holdout are listed too), the fixes and the limitations. To reproduce:

```bash
uv run agentic-rag eval                                         # full graph, with the model and judge of OLLAMA_MODEL
uv run agentic-rag eval --target node --node analyze_request     # routing alone
OLLAMA_MODEL=qwen3.5:4b OLLAMA_REASONING=false uv run agentic-rag eval --judge-model qwen2.5:7b-instruct
uv run agentic-rag eval --dataset data/eval/holdout.jsonl --judge-model qwen2.5:7b-instruct   # the holdout
```

### Load test & bottleneck analysis

**Method:** `agentic-rag loadtest` sends the questions of the evaluation set in turn to the compiled graph, from `--concurrency` threads that each call `graph.invoke`; three warm-up requests run first and are reported apart. Every run writes a JSON report and a Markdown summary to `data/eval/results/`: latency (mean, min, p50, p95, p99, max), throughput, error rate and the latency of every node.

**Results** (RTX 5070 Laptop GPU, Ollama with its defaults unless noted, 100 requests unless noted, no errors in any run):

| Run | Concurrency | Throughput | p50 | p95 |
|---|---|---|---|---|
| Fake LLM and embeddings (the framework alone) | 4 | 4 520 / min | 0.01 s | 0.06 s |
| `qwen2.5:7b-instruct` | 4 | 9.5 / min | 20.2 s | 58.0 s |
| `qwen3.5:4b`, thinking off (default) | 4 | 12.2 / min | 17.6 s | 35.4 s |
| `qwen3.5:4b`, one request at a time (50 requests) | 1 | 11.5 / min | 3.6 s | 14.9 s |
| `qwen2.5:7b-instruct`, Ollama with 4 parallel slots | 4 | 20.0 / min | 10.3 s | 25.0 s |
| `qwen3.5:4b` without the LLM grading | 4 | 10.7 / min | 18.2 s | 41.4 s |
| `qwen3.5:4b`, **RAG reliability update** | 4 | 12.0 / min | 20.4 s | 39.7 s |
| `qwen3.5:4b`, **RAG reliability update**, one request at a time (50 requests) | 1 | 11.7 / min | 3.9 s | 12.9 s |

**Bottleneck:** LLM inference on a server that runs one request at a time. A request makes 5.3 LLM calls on average, which take 99 % of its time; the retrieval takes 20 ms, the tools and the orchestration milliseconds. From one to four concurrent requests the throughput grows by only 6 % while the median latency grows fivefold, because every call waits in Ollama's queue; the GPU is busy all the time. The RAG reliability update leaves the bottleneck where it is: a request makes about 4.7 LLM calls instead of 5.3, because exact tool answers skip the answer and its verification, but the grading call reads up to 20 candidates instead of 4. For one user the median is 3.9 s (3.6 s before) and the p95 12.9 s (14.9 s); with four, the throughput is unchanged (12.0 against 12.2 per minute) while the median grows by 16 % (20.4 s against 17.6 s), because the longer prompts wait in the same queue.

**Proposals:**

1. **Parallel decoding slots with a model that batches** (measured): `OLLAMA_NUM_PARALLEL=4` doubles the throughput of the 7B transformer (20.0 against 9.5 per minute) and cuts its p95 by 57 %, at the cost of VRAM; Ollama does not batch Qwen3.5's hybrid architecture, so for it the setting changes nothing. A deployment for several users should therefore choose the model together with the serving stack (parallel slots, or continuous batching on a larger GPU).
2. **Fewer and shorter calls on the critical path:** writing the answer is half of the service time, the verification and its re-plans a fifth. Skipping the verification of exact tool answers is now built and measured: 0.86 verifications per request instead of 1.14, and the tool questions of the evaluation take about 7 s instead of 9.5 s. Streaming the answer to the UI is still open. The candidate pool is the new lever: a smaller `RETRIEVAL_CANDIDATES` shortens the grading prompt (not measured yet; 4 restores the old latency but also the missed page). Dropping the relevance grading does not help: it was measured slower (10.7 against 12.2 per minute), because more chunks reach the answer prompt.

[docs/performance.md](docs/performance.md) has the per-node breakdown, the micro-benchmark of the parallel slots and the commands. To reproduce the main run:

```bash
uv run agentic-rag loadtest --requests 100 --concurrency 4
```

## Getting started

### Prerequisites

- Git.
- [uv](https://docs.astral.sh/uv/getting-started/installation/) for local development; it also installs Python 3.12 when it is missing.
- Docker with Docker Compose 2.24 or newer for the containers (`compose.yaml` uses the optional `env_file` syntax).
- For real answers, a local LLM served by [Ollama](https://ollama.com/): the Compose service, or Ollama installed on the host. The default model, `qwen3.5:4b` (4-bit, 3.4 GB, thinking off), fits in an 8 GB GPU with room to spare and also runs on the CPU, more slowly; with one request at a time it answered in 3.6 s at the median on an RTX 5070 Laptop GPU. Measured with the former default, the 7B model, in the Compose stack: on the CPU the `ollama` container used 7.7 GB of RAM and answered in 20–60 s; with the GPU override, warm answers took 3–10 s. Fake mode needs neither a model nor a GPU.
- Disk space for the full stack: the application image (3.02 GB measured), the Ollama image (9.3 GB), the chat model (3.4 GB), the embedding model and the corpus with its index (about 640 MB together).

### What works

Checked end to end on the development machine (Windows 11, 24-core CPU, RTX 5070 Laptop GPU):

- the tests pass offline, with the fake LLM and the fake embeddings;
- the CLI lists its commands and `config` prints the effective settings;
- `ingest --download` downloads the corpus and builds the vector index, and a plain `ingest` keeps the index in step with the corpus. Measured on the development machine (24-core CPU): the download takes about 25 s, the first build about 6 minutes (embedding the 18 654 chunks with the default model on the CPU), and a repeated `ingest` 8 s, because unchanged chunks are not embedded again. A query against the index takes about 10 ms after the model has loaded (about 11 s);
- the RAG subgraph answers `invoke({"query": ...})` with a cited context and its sources. In fake mode it skips the model calls; with Ollama it rewrites and grades (measured with Qwen2.5-7B-Instruct, the former default, on a laptop GPU, see *Design decisions*). A question outside the corpus, such as *What is the capital of France?*, gets an empty context;
- `eval` runs the question set through the graph or one node, and `loadtest` sends it under load; both write a JSON report and its Markdown summary and print the summary;
- the main workflow answers: in fake mode with scripted replies that walk every route (a greeting, one search, two parallel searches, a tool call), with Ollama with real answers. With the default `qwen3.5:4b`, one request at a time, the median is 3.6 s (load test); measured with Qwen2.5-7B-Instruct, the former default, warm: 0.5–3 s for a direct reply, 2–9 s for a tool question, 8–30 s for a question that needs searches (the first search of a process also loads the embedding model, about 16 s); the details are in [docs/architecture.md](docs/architecture.md#measured-with-ollama);
- the Streamlit UI streams the main workflow: the step panel shows each step as it finishes, groups the parallel ones by LangGraph step and lists under each search the steps of the RAG subgraph (the English search query, the retrieved and the kept chunks), and the retrieved-context panel shows the numbered sources with a link to each page. The empty chat offers one example question per route (a search, a comparison asked in Hungarian, one question per tool), which also work with the fake LLM. A missing index, an index built with other embeddings, an unreachable Ollama and a model that is not pulled are explained in the chat with the fix. A run the user stops gets the turn *Stopped before an answer was produced.*, the agent receives the new question with only the earlier questions that were answered, `$` signs in answers are shown as text (no LaTeX), and invalid settings or an unreadable `.env` replace the chat with an *Invalid configuration* error;
- from a fresh clone, `docker compose up --build` pulls the chat model, downloads the corpus, builds the index and serves the UI with no other step (measured: 25 minutes for the first start on the development machine, 11 s for a restart); in fake mode a single container is ready in about 45 s.

### Local development with uv

Install uv (other options are in its [installation guide](https://docs.astral.sh/uv/getting-started/installation/)):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh      # Linux and macOS
```

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows
```

Clone the repository and set it up:

```bash
git clone https://github.com/Csabikahhh/agentic-rag-chatbot-poc.git
cd agentic-rag-chatbot-poc
uv sync --locked                                 # .venv with Python 3.12, the locked dependencies and the dev tools
uv run pytest                                    # offline test suite; ends with "... passed, 1 deselected"
uv run ruff check .                              # lint
uv run ruff format --check .                     # formatting
uv run python -m agentic_rag --help              # the commands (or: uv run agentic-rag --help)
uv run python -m agentic_rag config              # the effective settings
uv run agentic-rag ingest --download             # download the corpus and build the vector index
uv run streamlit run src/agentic_rag/ui/app.py   # the UI on http://localhost:8501
```

**The knowledge base.** `ingest --download` needs git and network access. It fetches each source of [`data/sources.toml`](data/sources.toml) at its pinned commit into `data/raw/<id>/` (gitignored), skipping sources that are already up to date, and then builds the index in `data/chroma_db/`. Without `--download`, `ingest` only updates the index from the corpus that is in place: it embeds new and changed chunks and removes the chunks of deleted pages. After changing `EMBEDDING_PROVIDER` or `EMBEDDING_MODEL`, rebuild it with `ingest --rebuild`. To change the corpus, edit `data/sources.toml` (a new commit, other patterns or a new source) and run `ingest --download` again.

**Fake mode** runs without Ollama and without model downloads: the scripted fake LLM (`LLM_PROVIDER=fake`) and the offline hashing embeddings (`EMBEDDING_PROVIDER=fake`). Set the two variables in the shell, or put them in `.env` (see [Configuration](#configuration)):

```bash
LLM_PROVIDER=fake EMBEDDING_PROVIDER=fake uv run streamlit run src/agentic_rag/ui/app.py
```

```powershell
$env:LLM_PROVIDER="fake"; $env:EMBEDDING_PROVIDER="fake"; uv run streamlit run src/agentic_rag/ui/app.py
```

`uv sync --locked` stops with an error instead of rewriting `uv.lock` when the lock file is out of date with `pyproject.toml`; the image build uses the same check.

The tests run in fake mode and ignore the shell's settings and `.env`. The one exception is the live Ollama test (marker `ollama`): plain `uv run pytest` deselects it, so the summary reads `807 passed, 1 deselected` (measured on 2026-10-03). The download tests fetch from a git repository created in a temporary directory and are skipped when git is not installed. `uv run pytest -m ollama` runs it, as shown below.

**CI.** [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs the same checks on every pull request and every push to `main`: `uv sync --locked`, `ruff check`, `ruff format --check` and `pytest` in fake mode, then `docker build` and the image's `agentic-rag --version`. No model, GPU or corpus is needed.

**Ollama on the host** is the fastest loop with a real model. Install [Ollama](https://ollama.com/download), start it (the desktop app, or `ollama serve`) and pull the model; the default `OLLAMA_BASE_URL` (`http://localhost:11434`) reaches it:

```bash
ollama pull qwen3.5:4b
uv run pytest -m ollama       # live check against the local server; skipped when it is not reachable
```

The live test follows `OLLAMA_BASE_URL` and `OLLAMA_MODEL` from the shell, so it can also check another server or model.

### Run with Docker Compose

The brief asks for a `docker-compose.yml`; the repository provides it as `compose.yaml`, the file name the Compose documentation prefers, which `docker compose` picks up automatically.

| Service | Image | Role |
|---|---|---|
| `app` | Built from the `Dockerfile` | Streamlit UI on <http://localhost:8501> (published on 127.0.0.1 only); runs as a non-root user |
| `ollama` | `ollama/ollama:0.35.0` | Serves the LLM; reachable as `http://ollama:11434` inside the stack, not published on the host |
| `ollama-pull` | `ollama/ollama:0.35.0` | One-shot: pulls `OLLAMA_MODEL` unless the `ollama-data` volume already has it |

The named volumes keep the downloads and the index across rebuilds: `ollama-data` (Ollama models), `corpus-data` (the corpus), `chroma-data` (the vector index, one directory per embedding provider, so switching between the full stack and fake mode does not rebuild it) and `hf-cache` (Hugging Face models). No host directory is mounted.

**The knowledge base in the container.** The app's command, `agentic-rag serve`, prepares the knowledge base before it starts the UI (`INGEST_ON_START=true`, the default): it downloads the corpus sources that the `corpus-data` volume does not hold yet (with git, at the pinned commits of `data/sources.toml`), then builds the index or brings it up to date with the corpus; an index built with other embeddings is rebuilt. Nothing has to be prepared on the host, and a restart checks the corpus and the index in a few seconds. To rebuild the index from scratch:

```bash
docker compose run --rm --no-deps app agentic-rag ingest --rebuild
```

**Full stack:**

```bash
docker compose up --build
```

The first start builds the image (about a minute with a warm cache, several minutes without), downloads the Ollama image and the chat model (several GB), then the app downloads the corpus (about 20 s) and the embedding model and embeds the 18 654 chunks on the CPU (about 6 minutes, the embedding model download included) before the UI starts. Measured on the development machine: 25 minutes from `docker compose up --build` to a healthy app, most of it the downloads (19 minutes for the Ollama image and the model at about 7 MB/s). Meanwhile the app container reports `health: starting`; `docker compose logs -f app` follows the progress. Later starts reuse the volumes: the UI answers 11 s after `docker compose up`.

**Fake mode** (only the `app` service; no Ollama, no model downloads):

```bash
LLM_PROVIDER=fake EMBEDDING_PROVIDER=fake docker compose up --build --no-deps app
```

```powershell
$env:LLM_PROVIDER="fake"; $env:EMBEDDING_PROVIDER="fake"; docker compose up --build --no-deps app
```

`--no-deps` leaves out the two Ollama services. The corpus is still downloaded, and the index of the fake embeddings is built in about 20 s, so the UI answers about 45 s after the first start. In PowerShell the variables stay set for the rest of the session; setting them in `.env` works as well.

**NVIDIA GPU for Ollama** (an optional override file):

```bash
docker compose -f compose.yaml -f compose.gpu.yaml up --build
```

It needs an NVIDIA driver and GPU support in Docker: Docker Desktop with the WSL 2 backend on Windows, or the NVIDIA Container Toolkit on Linux. To make it the default, set `COMPOSE_FILE=compose.yaml:compose.gpu.yaml` in `.env` (with `;` as the separator on Windows). Without the override, Ollama runs on the CPU. Measured with the former default, the 7B model: 20–60 s per answer on the CPU, 3–10 s warm on an RTX 5070 Laptop GPU (the first answer also loads the models, about 45 s).

**Ollama on the host** instead of the `ollama` service:

```bash
docker compose run --rm --no-deps --service-ports -e OLLAMA_BASE_URL=http://host.docker.internal:11434 app
```

**CLI commands in the container:**

```bash
docker compose run --rm --no-deps app agentic-rag config
```

Run `eval` and `loadtest` on the host (`uv run agentic-rag eval`, `uv run agentic-rag loadtest`); this is the recommended way. The stack mounts no host directory, so running them in the container needs a `./data/eval:/app/data/eval` bind mount, and the reports can then only be written if `data/eval/results` is writable by the container's user.

**Linux hosts and UID 10001.** The `app` container runs as UID and GID 10001. A bind mount keeps the owner of the host directory, so on a Linux engine the app can write to a bind mount only if that directory is writable for UID 10001; Docker Desktop on Windows and macOS hides this, because it presents bind mounts as writable for everyone. Either make the directory writable for UID 10001, or build the image with your own IDs through the `APP_UID` and `APP_GID` build arguments:

```bash
APP_UID=$(id -u) APP_GID=$(id -g) docker compose up --build
```

Named volumes (`corpus-data`, `chroma-data`, `hf-cache`) take their owner from the image only while they are empty, so after changing the IDs, change their owner in place (the command is in the comment on the `app` service in [`compose.yaml`](compose.yaml)) or recreate them with `docker compose down -v`, which also deletes the corpus, the index and the downloaded models.

**The image on its own** (the required `Dockerfile`, without Compose):

```bash
docker build -t agentic-rag-chatbot:dev .
docker run --rm -p 127.0.0.1:8501:8501 -e LLM_PROVIDER=fake -e EMBEDDING_PROVIDER=fake agentic-rag-chatbot:dev
```

Every new container downloads the corpus and builds the index again (about 45 s in fake mode). To keep them, add two volumes: `--mount type=volume,source=agentic-rag-corpus,target=/app/data/raw --mount type=volume,source=agentic-rag-index,target=/app/data/chroma_db`. `docker stop` ends the container at once, also while it is preparing the knowledge base. With plain `docker build`, pass `--build-arg APP_UID=... --build-arg APP_GID=...` for other IDs.

**Image layers.** The `Dockerfile` has two stages. The `deps` stage installs only the dependencies pinned in `uv.lock` (`uv sync --locked --no-dev --no-install-project`); `--locked` stops the build when `uv.lock` is out of date with `pyproject.toml`. The runtime stage copies that virtual environment (1.71 GB) and compiles its bytecode (415 MB) in two layers that do not depend on the code, then adds `src/` (348 kB) and a small editable install of the project (115 kB). A change to `src/` therefore rebuilds only the two small layers: measured at 7 s, against about 53 s and a new 2.11 GB layer before the split. The image is 3.02 GB (`python:3.12.14-slim-trixie`, CPU-only torch, and git for the corpus download, 105 MB); it contains no uv, no build files and no dev dependencies, and code and dependencies are root-owned and read-only for the app user.

**Stop and clean up:**

```bash
docker compose down      # remove the containers and the network; keep the volumes
docker compose down -v   # also delete the volumes
```

> **Warning:** `docker compose down -v` deletes the downloaded models (`ollama-data`, `hf-cache`), the corpus (`corpus-data`) and the vector index (`chroma-data`); the next start downloads and builds them again.

More options, such as publishing the Ollama API on the host through a local `compose.override.yaml`, are described in the header of [`compose.yaml`](compose.yaml).

> Verified: the image build (also with `APP_UID`/`APP_GID` set, and the `--locked` failure on a stale lock file), the layer reuse after a change to `src/`, `docker build .` on its own, both Compose configurations, a simulated Linux bind mount owned by UID 1000, and on Windows with Docker Desktop: the full stack from a fresh clone (healthy after 25 minutes, then real answers from the container), a restart, fake mode with `--no-deps` and a plain `docker run` (the corpus downloaded and the index built at start-up, the UI answering), `docker stop` during the start-up preparation (stopped at once, exit code 130), and the GPU override (all 29 layers on an NVIDIA RTX 5070 Laptop GPU). Not run yet: a native Linux host and macOS. From Git Bash, `docker compose run -e NAME=/path` needs `MSYS_NO_PATHCONV=1`, or Git Bash rewrites the path into a Windows path.

### Configuration

All settings are environment variables, read by `agentic_rag.config.Settings`. [`.env.example`](.env.example) lists every variable with its default and a comment; copy it to `.env` (gitignored) and change what you need:

```bash
cp .env.example .env    # PowerShell: Copy-Item .env.example .env
```

Real environment variables take precedence over `.env`, and an empty value (`KEY=`) means the default. Keep `.env` in UTF-8: Windows PowerShell 5.1 writes UTF-16 with `>` and `Out-File`, so copy the file with `Copy-Item` as above. `uv run agentic-rag config` prints the effective values; an invalid value, or a `.env` that cannot be read or is not UTF-8, stops the CLI with exit code 2 and names the problem, and the UI shows it instead of the chat.

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama`, or `fake` for the scripted offline model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | URL of the Ollama server |
| `OLLAMA_MODEL` | `qwen3.5:4b` | Ollama chat model tag (decision 4) |
| `OLLAMA_NUM_CTX` | `8192` | Context window in tokens, 512–131072, sent as Ollama's `num_ctx`; the prompt and the answer share it, and Ollama silently truncates a longer prompt |
| `OLLAMA_TIMEOUT_S` | `120.0` | HTTP timeout in seconds of each Ollama request, greater than 0 |
| `OLLAMA_REASONING` | `false` | Thinking mode of reasoning models such as Qwen3.5: `false` turns it off, `true` on; models without one ignore it. Thinking made `qwen3.5:4b` about ten times slower in the evaluation, without better answers |
| `LLM_TEMPERATURE` | `0.0` | Sampling temperature, 0.0–2.0 |
| `EMBEDDING_PROVIDER` | `huggingface` | `huggingface`, or `fake` for the offline hashing embeddings |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | Hugging Face embedding model (provisional) |
| `DATA_DIR` | `data/raw` | Corpus directory |
| `CHROMA_DIR` | `data/chroma_db` | Vector index directory |
| `CHROMA_COLLECTION` | `documents` | Chroma collection name: 3–63 characters (a deliberate project limit), not an IPv4 address |
| `TOP_K` | `4` | Chunks retrieved per query |
| `GRADE_WITH_LLM` | `true` | Let the chat model drop retrieved chunks that do not help answer the query (one extra LLM call per retrieval; no effect in fake mode) |
| `MAX_RETRIES` | `2` | Bound on the verify → re-plan loop |
| `INGEST_ON_START` | `true` | At start-up (`agentic-rag serve`, the container's command), download the missing corpus sources and bring the index up to date; an index built with other embeddings is rebuilt |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` or `ERROR` |

In the Compose stack, `compose.yaml` sets `OLLAMA_BASE_URL=http://ollama:11434` for the `app` service, so a value in `.env` does not change it, and passes `LLM_PROVIDER`, `EMBEDDING_PROVIDER` and `OLLAMA_MODEL` from the shell or `.env`. Every other variable reaches the container only through `.env`. The full reference, with the validation rules, is in [docs/architecture.md](docs/architecture.md#configuration-reference).

### Command-line interface

`agentic-rag <command>` (or `python -m agentic_rag <command>`); `--help` shows the options of every command.

| Command | Purpose | Available |
|---|---|---|
| `config` | Print the effective settings as `KEY=value` lines | Now |
| `serve [--address HOST] [--port PORT]` | Start the UI; with `INGEST_ON_START`, first download the missing corpus and update the index (the container's command) | Now |
| `ingest [--rebuild] [--download] [--sources PATH]` | Build or update the vector index from `DATA_DIR`; with `--download`, first download the corpus sources (needs git) | Now |
| `export-graph [--graph {all,agent,rag}] [--format {markdown,mermaid}] [--output PATH]` | Mermaid diagrams of the compiled graphs | Now |
| `eval [--target {graph,node}] [--node NAME] [--dataset PATH] [--output-dir PATH] [--judge-model NAME]` | Functional evaluation: writes a JSON report and its Markdown summary, prints the summary | Now |
| `loadtest [--requests N] [--concurrency C] [--warmup W] [--output-dir PATH] [--dataset PATH]` | Load test against the compiled graph: writes a JSON report and its Markdown summary, prints the summary | Now |

Exit codes:

- 0 on success;
- 1 when the command failed: a feature planned for a later phase (`PlannedFeatureError`), a missing corpus and a failed download (`ingest`), and a missing or invalid question set or a missing index (`eval`, `loadtest`) print only their message, any other error its traceback;
- 2 for usage and configuration errors: invalid arguments (including an `InvalidArgumentError`, such as `eval --node` outside `NODE_TARGETS`), invalid settings, or a `ConfigurationError` (a `.env` that cannot be read or is not UTF-8, an invalid `data/sources.toml`, or an index built with other embeddings, `EmbeddingMismatchError`);
- 130 when interrupted.

The details are in [docs/architecture.md](docs/architecture.md#errors-and-exit-codes).


## Repository structure

```text
agentic-rag-chatbot-poc/
├── .claude/                        # Claude Code agents and skills used while building the project
├── .github/workflows/ci.yml        # CI: lint, format check, offline tests, image build
├── .streamlit/config.toml          # the UI theme (PwC-inspired colours, Georgia and Arial); no secrets
├── data/
│   ├── README.md                   # data layout, corpus rules, when to rebuild the index
│   ├── sources.toml                # the corpus sources: repositories, pinned commits, patterns, licenses
│   ├── raw/                        # downloaded corpus (DATA_DIR), one directory per source, and the
│   │                               #   browser-compat-data of the browser_support tool; gitignored
│   └── eval/
│       ├── README.md               # evaluation question-set schema and report formats
│       ├── questions.jsonl         # the development set: 17 questions
│       ├── holdout.jsonl           # the holdout: 24 questions kept out of development
│       └── results/                # committed evaluation and load-test reports (JSON and Markdown)
├── docs/
│   ├── architecture.md             # target graphs, state and cross-cutting contracts, configuration reference
│   ├── evaluation.md               # functional evaluation: method, results, model comparison, conclusions
│   ├── performance.md              # load test: results, per-node breakdown, bottleneck, proposals
│   ├── rag-improvements.md         # the RAG reliability update: changes, measurements, migration
│   ├── developer-guide.en.md       # developer guide, one chapter per page (source of the PDF)
│   ├── developer-guide.hu.md       # the developer guide in Hungarian
│   ├── build_developer_pdfs.py     # builds both PDFs (ReportLab; document tooling only)
│   ├── project-structure-plan.md   # repository plan and build order
│   └── project-structure-plan.hu.md  # the plan in Hungarian
├── output/pdf/                     # developer-guide-en.pdf and developer-guide-hu.pdf, built from docs/
├── src/
│   └── agentic_rag/
│       ├── __init__.py             # package version
│       ├── __main__.py             # `python -m agentic_rag`
│       ├── cli.py                  # commands: ingest · eval · loadtest · export-graph · config · serve
│       ├── config.py               # Settings from environment variables and .env; logging set-up
│       ├── errors.py               # PlannedFeatureError, ConfigurationError, InvalidArgumentError, planned()
│       ├── llm.py                  # chat model factory: Ollama, or the scripted fake
│       ├── embeddings.py           # embedding factory: sentence-transformers, or offline hashing
│       ├── tracing.py              # TraceEvent and the @traced node decorator
│       ├── reports.py              # RESULTS_DIR; RunReport, the base of EvalReport and LoadTestReport
│       ├── agent/                  # main agentic workflow
│       │   ├── types.py            # Intent, Verdict, SubtaskKind, without LangGraph
│       │   ├── state.py            # AgentState, Subtask, SubtaskResult
│       │   ├── prompts.py          # the four prompts, their output schemas, the tool catalog
│       │   ├── nodes.py            # the seven node functions
│       │   ├── routing.py          # conditional edges and the Send fan-out
│       │   ├── tools.py            # search_knowledge_base and the three non-retrieval tools
│       │   ├── contrast.py         # WCAG 2.2 contrast ratio and verdicts
│       │   ├── specificity.py      # Selectors Level 4 specificity
│       │   ├── compat.py           # browser support from MDN browser-compat-data
│       │   └── graph.py            # NODE_NAMES and build_agent_graph()
│       ├── rag/                    # RAG subgraph: rewrite, retrieve, grade, build the cited context
│       │   ├── state.py            # RagState, RagInput, RagOutput, Source (implemented contracts)
│       │   ├── nodes.py            # rewrite_query · retrieve · grade_documents · build_context, prompts
│       │   ├── lexical.py          # BM25 keyword search (SQLite FTS5) and reciprocal-rank fusion
│       │   └── graph.py            # RAG_NODE_NAMES, MIN_SCORES and build_rag_graph()
│       ├── ingestion/              # ingestion pipeline: download, clean, chunk, embed and store
│       │   ├── sources.py          # the source list, glob patterns, manifests and page URLs
│       │   ├── download.py         # sparse git checkout of each source at its pinned commit
│       │   ├── markdown.py         # front matter, dialect cleaning (MDN, MDX, VitePress, MDC), sections
│       │   ├── loaders.py          # corpus files → one Document per section, with citation metadata
│       │   ├── chunking.py         # structure-aware chunks with a context line (900/150, code up to 1800)
│       │   ├── index.py            # build, update, open and check the Chroma index; IndexStats
│       │   └── prepare.py          # start-up preparation: download what is missing, update the index
│       ├── evaluation/
│       │   ├── dataset.py          # EvalItem and the questions.jsonl loader
│       │   ├── metrics.py          # hit@k, routing accuracy, the LLM-judged correctness and faithfulness
│       │   └── runner.py           # run_evaluation(), the report models and the Markdown summary
│       ├── loadtest/
│       │   └── runner.py           # run_load_test(), the latency statistics, the report and its summary
│       └── ui/
│           ├── app.py              # Streamlit entrypoint
│           └── components.py       # step panel, retrieved-context panel, failure hints, examples
├── task/                           # assignment brief (Hungarian); local only, gitignored
├── tests/                          # offline pytest suite (fake providers)
│   ├── conftest.py                 # keeps the shell and .env out of the tests; the `settings` fixture
│   ├── test_cli.py                 # commands, options and exit codes
│   ├── test_config.py              # defaults, environment and .env handling, validation, logging
│   ├── test_embeddings.py          # offline fake and Hugging Face branch, without downloads
│   ├── test_agent_graph.py         # main workflow: contract, routing, nodes, every route in fake mode
│   ├── test_evaluation.py          # dataset loader, metrics, judge, run_evaluation and the reports
│   ├── test_ingestion.py           # sources, download (local git repository), cleaning, chunking, index
│   ├── test_llm.py                 # provider selection, scripted fake; live Ollama check (marker `ollama`, deselected by default)
│   ├── test_loadtest.py            # percentiles, run_load_test on a small index, the report and its summary
│   ├── test_rag_subgraph.py        # RAG nodes with stand-ins; the compiled subgraph on a small index
│   ├── test_rag_improvements.py    # reliability update: keyword index and fusion, grading pool, tool fast
│   │                               #   path, verification fallback, evidence groups, citations, holdout
│   ├── test_state.py               # state contracts and reducers
│   ├── test_tools.py               # contrast, specificity, browser support and the tool layer
│   ├── test_tracing.py             # step-trace primitives
│   └── test_ui.py                  # Streamlit UI under AppTest
├── .dockerignore                   # build-context allowlist
├── .env.example                    # every setting with its default
├── .gitattributes                  # LF line endings in every checkout, Windows included
├── .gitignore
├── .python-version                 # 3.12
├── compose.gpu.yaml                # optional NVIDIA GPU override for the ollama service
├── compose.yaml                    # app + ollama + one-shot model pull
├── Dockerfile                      # deps and runtime stages, uv sync --locked, non-root runtime, health check
├── LICENSE
├── pyproject.toml                  # dependencies, console script, ruff and pytest settings
├── README.md                       # documentation (English)
├── README.hu.md                    # documentation (Hungarian)
└── uv.lock                         # locked dependency versions
```

Every package directory also has an `__init__.py`. Generated and local-only paths are not shown: `.venv/`, the tool caches and `data/chroma_db/` (the vector index, created by `agentic-rag ingest`). [data/README.md](data/README.md) describes the data layout, the corpus rules and when to rebuild the index; [data/eval/README.md](data/eval/README.md) describes the evaluation formats.

The [build order](docs/project-structure-plan.md#8-build-order) of the plan records which phase added which part, and its [section 12](docs/project-structure-plan.md#12-deviations-from-the-plan) where the result differs from the plan.

## License

Released under the [MIT License](LICENSE). © 2026 Csaba Ovari

**Third-party notice:** `Dockerfile`, `.dockerignore` and `compose.yaml` contain portions adapted from the assets of the `docker-project-foundations` skill of [docker/skills](https://github.com/docker/skills) (the skill itself is in `.claude/skills/docker-project-foundations/`), which is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). Each of these files names its source in its header and states that it was modified; the adapted portions are used under the Apache License 2.0.
