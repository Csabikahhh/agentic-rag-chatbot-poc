# Project structure plan

**English** | [Magyar](project-structure-plan.hu.md)

> **Status:** proposal, written 2026-10-01. This document plans the *skeleton* of the repository and the order in which to build it. Detailed design (exact prompts, metrics, chunk sizes) is decided while building and recorded in the [README](../README.md) and the other documents in `docs/`.
>
> **Progress:** Phase 1 done on 2026-10-01. Brought forward with it: the container setup of Phase 6 (`Dockerfile`, `compose.yaml`, `compose.gpu.yaml`) and the shared modules of section 5.7 (`config.py`, `llm.py`, `embeddings.py`, `tracing.py`), together with the state contracts, the Streamlit shell and typed skeletons for Phases 2–8. The foundation was then reviewed and revised on the same day; those changes are part of section 12 as well. Next: decisions 8–9, then Phase 2. Details are in [section 8](#8-build-order); the deviations from this plan are recorded in [section 12](#12-deviations-from-the-plan).

## Contents

1. [Goal and scope](#1-goal-and-scope)
2. [Starting point](#2-starting-point)
3. [Decisions to lock in before scaffolding](#3-decisions-to-lock-in-before-scaffolding)
4. [Target repository layout](#4-target-repository-layout)
5. [Module responsibilities](#5-module-responsibilities)
6. [Configuration and run modes](#6-configuration-and-run-modes)
7. [Containerization plan](#7-containerization-plan)
8. [Build order](#8-build-order)
9. [Requirement traceability](#9-requirement-traceability)
10. [Conventions](#10-conventions)
11. [Risks and open questions](#11-risks-and-open-questions)
12. [Deviations from the plan](#12-deviations-from-the-plan)

## 1. Goal and scope

The structure has to carry everything the assignment asks for, without growing into a production system:

- A **LangGraph main workflow** with at least 5 nodes, conditional routing, sub-task decomposition and shared state.
- A **modular RAG subgraph** that the main workflow calls and that is not counted towards the 5 nodes.
- **At least 2 tools**, one of which is not retrieval.
- A **local open-source LLM** (no paid APIs) with a **dummy fallback** for tests and for machines that cannot run a model.
- An **ingestion pipeline** for a small, well-processed text corpus and a persistent vector index.
- A **Streamlit UI** that shows the agent's steps and the retrieved context.
- A **Dockerfile** (required) and a **Compose stack** (UI + model server).
- A **functional evaluation** (10–20 questions) and a **load test** (50–200 queries) with per-node latency, bottleneck analysis and optimization proposals.
- **Bilingual documentation** (English + Hungarian) that stays in sync with the code.

Out of scope for the skeleton: authentication, multi-user persistence, cloud deployment, hosted tracing services, a public HTTP API (optional later, see [§11](#11-risks-and-open-questions)).

## 2. Starting point

Already in the repository:

| Item | Notes |
|---|---|
| `README.md`, `README.hu.md` | Bilingual documentation with the requirement checklist and `To be completed` placeholders |
| `.gitignore` | Already ignores `task/`, `.env`, `models/`, `chroma_db/`, `.chroma/`, `*.gguf`, `*.bin`, virtual environments and caches |
| `.claude/agents/`, `.claude/skills/` | Four Claude Code agents (LangGraph, RAG, Streamlit, Docker) with their skills |
| `docs/` | Empty, reserved for diagrams and reports |
| `task/` | The assignment brief (Hungarian), kept local only |
| `LICENSE` | MIT |

Local environment, checked on 2026-10-01:

| Resource | Value |
|---|---|
| OS | Windows 11 Pro |
| Python | 3.14.3 (no `uv`, no `poetry` installed) |
| Docker | 29.8.0, Compose v5.5.1 |
| Ollama | client 0.35.0 installed, server not running |
| RAM | 31 GB |
| CPU | Intel Core Ultra 9 275HX, 24 cores |
| GPU | NVIDIA GeForce RTX 5070 Laptop, 8 GB VRAM |

A 7–8B parameter model in 4-bit quantization (~4.5–5 GB) fits in the GPU with room for the KV cache, and a 3–4B model leaves headroom for running the embedding model on the GPU as well.

## 3. Decisions to lock in before scaffolding

The recommendations below are the defaults the skeleton is built around. Each final choice goes into the *Design decisions* table of the README together with its trade-off.

| # | Decision | Options | Recommendation | Why |
|---|---|---|---|---|
| 1 | Packaging and Python version | `pip` + `requirements.txt` / `poetry` / `uv` + `pyproject.toml` + `uv.lock` | **`uv`**, Python **3.12** pinned in `.python-version` and in the Dockerfile | A lock file makes the build reproducible, `uv` installs the pinned Python itself, and 3.12 has the widest wheel coverage for the ML stack (torch, chromadb). The local 3.14 is not needed. |
| 2 | Package layout | flat / `src` layout | **`src/agentic_rag/`** | Prevents accidental imports from the working directory; works cleanly in the container and in tests. |
| 3 | LLM serving | Ollama / `llama-cpp-python` in-process / dummy only | **Ollama** as a Compose service, with the host's Ollama usable in development, plus a **fake provider** | Ollama gives an HTTP API, GPU support and a clean multi-container story. `llama-cpp-python` compiles inside the image and slows builds. The fake provider satisfies the "dummy LLM" fallback and makes tests and CI model-free. |
| 4 | LLM model | 3–4B vs. 7–8B instruct models | Shortlist: `qwen2.5:7b-instruct`, `llama3.1:8b`, `gemma3:4b` (check the current Ollama tags); default to a **7B-class model at Q4** | Fits 8 GB VRAM. Choose by the language of the corpus (Hungarian text needs a model with strong multilingual coverage) and by the latency measured in the load test. Keep a 3B model as the fast alternative. |
| 5 | Embeddings | `sentence-transformers` in-process / Ollama embeddings / `fastembed` | **`sentence-transformers` via `langchain-huggingface`**, CPU-only torch wheels | Retrieval then works in fake mode without Ollama. Shortlist: `intfloat/multilingual-e5-small` for mixed-language corpora, `BAAI/bge-small-en-v1.5` for English only. Ollama embeddings are the fallback if the image size becomes a problem. |
| 6 | Vector store | Chroma / FAISS / in-memory | **Chroma**, persistent client, `data/chroma_db/` | Persistence plus metadata filtering, no pickle deserialization, already in `.gitignore`. |
| 7 | Tool-calling style | native `bind_tools` + `ToolNode` / structured-output planner + explicit tool nodes | **Structured-output planner + explicit nodes** | Small local models are unreliable at native tool calling. A planner that emits a JSON list of typed sub-tasks is robust and still demonstrates autonomous decisions. |
| 8 | Domain and corpus | — | **Your call**, before Phase 2 | Criteria: a real user need; a small corpus with a clear license (public domain, CC or your own); text that is stable enough for reference answers; a domain where a non-retrieval tool is natural. |
| 9 | Non-retrieval tool | calculator / date and deadline computation / unit conversion / structured table lookup | Depends on the domain; must be **deterministic and local** | Examples: deadline computation for regulation Q&A, a net/gross calculator for payroll rules, a unit converter for technical manuals. |
| 10 | HTTP API layer | none / FastAPI service | **None in the skeleton** | The load test drives the compiled graph directly, which gives a cleaner per-node breakdown. A FastAPI service can be added later as a third Compose component. |

## 4. Target repository layout

```text
agentic-rag-chatbot-poc/
├── .claude/                          # Claude Code agents and skills (present)
├── .github/
│   └── workflows/
│       └── ci.yml                    # optional: ruff + pytest in fake mode + docker build
├── data/
│   ├── raw/                          # source corpus (small, license checked) – or fetched by `ingest --download`
│   ├── eval/
│   │   ├── questions.jsonl           # 10–20 evaluation questions with reference answers and expected sources
│   │   └── results/                  # committed outputs of the final evaluation and load-test runs
│   └── chroma_db/                    # built vector index (gitignored)
├── docs/
│   ├── project-structure-plan.md     # this plan
│   ├── project-structure-plan.hu.md  # this plan, Hungarian
│   ├── architecture.md               # Mermaid export of both graphs, state schema, node and tool tables
│   ├── evaluation.md                 # functional evaluation: method, results, conclusions
│   └── performance.md                # load test: setup, metrics, bottleneck, proposals
├── src/
│   └── agentic_rag/
│       ├── __init__.py
│       ├── __main__.py               # `python -m agentic_rag <command>`
│       ├── cli.py                    # ingest · eval · loadtest · export-graph
│       ├── config.py                 # pydantic-settings: provider, model names, paths, top_k
│       ├── llm.py                    # LLM factory: ollama | fake
│       ├── embeddings.py             # embedding model factory
│       ├── tracing.py                # step-trace collection shared by UI, evaluation and load test
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── loaders.py            # PDF / Markdown / text → Documents with metadata
│       │   ├── chunking.py           # splitter configuration
│       │   └── index.py              # build and load the Chroma index
│       ├── rag/                      # RAG subgraph (not counted towards the 5 nodes)
│       │   ├── __init__.py
│       │   ├── state.py
│       │   ├── nodes.py              # rewrite_query · retrieve · grade_documents · build_context
│       │   └── graph.py              # compiled `rag_graph`
│       ├── agent/                    # main agentic workflow
│       │   ├── __init__.py
│       │   ├── state.py              # AgentState with reducers
│       │   ├── nodes.py              # the 7 main nodes
│       │   ├── routing.py            # conditional-edge functions, Send fan-out
│       │   ├── tools.py              # search_knowledge_base + the non-retrieval tool(s)
│       │   └── graph.py              # StateGraph wiring, compiled `agent_graph`
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── dataset.py            # load questions.jsonl
│       │   ├── metrics.py            # correctness, faithfulness, hit@k, routing accuracy
│       │   └── runner.py
│       ├── loadtest/
│       │   ├── __init__.py
│       │   └── runner.py             # async N queries × concurrency, percentiles, per-node latency
│       └── ui/
│           ├── app.py                # Streamlit entrypoint
│           └── components.py         # agent-step trace panel, retrieved-sources panel
├── tests/
│   ├── conftest.py                   # fake LLM, tiny fixture corpus, temporary Chroma directory
│   ├── test_ingestion.py
│   ├── test_rag_subgraph.py
│   ├── test_agent_graph.py           # routing, decomposition, node count ≥ 5, bounded retry loop
│   ├── test_tools.py
│   └── test_ui_smoke.py              # streamlit.testing.v1.AppTest
├── .dockerignore
├── .env.example
├── .gitignore                        # present
├── .python-version                   # 3.12
├── compose.yaml                      # app + ollama + one-shot model pull
├── Dockerfile                        # multi-stage, uv-based, non-root
├── LICENSE                           # present
├── pyproject.toml                    # dependencies, ruff, pytest, console entrypoint
├── README.md · README.hu.md          # present, updated per phase
├── task/                             # present, gitignored
└── uv.lock
```

Naming note: the brief says `docker-compose.yml`; the Docker documentation and the project's Docker skill use `compose.yaml`. `docker compose` picks up both automatically, so the repository uses `compose.yaml` and the README mentions the equivalence.

## 5. Module responsibilities

### 5.1 Main agentic workflow (`agent/`)

Seven nodes, so the 5-node minimum still holds if two are merged later.

| # | Node | Kind | Reads | Writes |
|---|---|---|---|---|
| 1 | `analyze_request` | LLM | `messages` | `question`, `intent` (`direct` / `single` / `complex` / `tool`) |
| 2 | `plan_subtasks` | LLM | `question`, previous `verdict` | `subtasks` – typed list: `{kind: retrieve \| tool, input}` |
| 3 | `run_rag_subtask` | subgraph call | one sub-task | appends to `subtask_results` (context, sources, scores) |
| 4 | `call_tool` | action | one sub-task | appends to `subtask_results` (tool output) |
| 5 | `synthesize_answer` | LLM | `subtask_results` | `draft_answer` |
| 6 | `verify_answer` | LLM | `draft_answer`, contexts | `verdict` (`grounded` / `insufficient`), `retry_count` |
| 7 | `finalize_response` | data | `draft_answer`, sources | `answer`, `sources`, final `trace` entry, assistant message |

Routing rules (conditional edges):

- After `analyze_request`: `direct` → `finalize_response`; `single` → one `Send` to `run_rag_subtask`; `complex` → `plan_subtasks`; `tool` → one `Send` to `call_tool`.
- After `plan_subtasks`: one `Send` per sub-task, dispatched to `run_rag_subtask` or `call_tool` by `kind`. This is the decomposition and independent execution: LangGraph runs the sends in parallel and fans in before `synthesize_answer`.
- After `verify_answer`: `grounded` → `finalize_response`; `insufficient` with `retry_count < 2` → `plan_subtasks` (re-plan with the critique); otherwise `finalize_response` with an explicit "partially answered" note.

```mermaid
flowchart TD
    S((start)) --> A[analyze_request]
    A -->|direct| F[finalize_response]
    A -->|single| R[run_rag_subtask]
    A -->|complex| P[plan_subtasks]
    A -->|tool| T[call_tool]
    P -->|Send per sub-task| R
    P -->|Send per sub-task| T
    R --> Y[synthesize_answer]
    T --> Y
    Y --> V[verify_answer]
    V -->|grounded| F
    V -->|insufficient, retries left| P
    V -->|retries exhausted| F
    F --> E((end))
    R -. invokes .-> G[[RAG subgraph]]
```

Tools (`agent/tools.py`):

- `search_knowledge_base(query)` – wraps `rag_graph.invoke`; the only retrieval tool.
- One non-retrieval tool chosen together with the domain (decision 9), deterministic, with unit tests. Both are exposed as `@tool` functions so they can also be bound to a tool-calling model later without restructuring.

### 5.2 RAG subgraph (`rag/`)

Own `RagState` (`query`, `rewritten_query`, `documents`, `scores`, `context`, `sources`). Four nodes:

| Node | Role |
|---|---|
| `rewrite_query` | Optional LLM rewrite for retrieval; skipped in fake mode |
| `retrieve` | Top-k similarity search with scores and metadata from Chroma |
| `grade_documents` | Drop irrelevant chunks (score threshold, optionally an LLM grade) |
| `build_context` | De-duplicate, order, format with citation markers |

The main graph calls it from `run_rag_subtask` with an explicit input/output mapping, so the two state schemas stay independent and the subgraph can be tested and load-tested on its own.

### 5.3 State (`agent/state.py`)

```python
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    intent: Literal["direct", "single", "complex", "tool"]
    subtasks: list[Subtask]
    subtask_results: Annotated[list[SubtaskResult], operator.add]   # fan-in from Send
    draft_answer: str
    verdict: Literal["grounded", "insufficient"] | None
    retry_count: int
    answer: str
    sources: list[Source]
    trace: Annotated[list[TraceEvent], operator.add]                # consumed by UI and load test
```

Raw data lives in state; prompts are formatted inside the nodes.

### 5.4 Ingestion (`ingestion/`)

Load → split → embed → store. Loaders attach `source`, `title` and `page` / `section` metadata so the UI can cite. Chunking starts at ~800–1000 characters with 10–20 % overlap and is tuned against the evaluation set. `index.py` exposes `build_index()` and `load_index()`; the CLI `ingest` command is idempotent, and the container entrypoint can call it when the index directory is empty.

### 5.5 UI (`ui/`)

One Streamlit entrypoint. Chat with `st.chat_message` / `st.chat_input`; a live step panel fed by `agent_graph.stream(..., stream_mode="updates")`; an expander with the retrieved chunks, their sources and scores; a sidebar for provider, model and top-k. Must run in fake mode without Ollama.

### 5.6 Evaluation and load test (`evaluation/`, `loadtest/`)

- `evaluation/`: reads `data/eval/questions.jsonl` (`question`, `reference_answer`, `expected_documents`, `expected_intent`), runs either a single node or the full graph, and scores correctness (local LLM-as-judge or semantic similarity), faithfulness, retrieval hit@k and routing accuracy. Writes JSON to `data/eval/results/` and a summary to `docs/evaluation.md`.
- `loadtest/`: async harness that sends N queries (50–200) at a given concurrency against the compiled graph, records per-request latency and per-node latency from the trace, and reports mean / p50 / p95 / p99 / max, throughput and error rate. Running it in both `fake` and `ollama` modes isolates the LLM's share of the latency, which is the core of the bottleneck analysis.

### 5.7 Shared modules (`config.py`, `llm.py`, `embeddings.py`, `tracing.py`)

- `config.py`: one `Settings` class (pydantic-settings) read from the environment and `.env`.
- `llm.py`: `get_chat_model()` returns `ChatOllama` or a scripted fake chat model whose replies depend on the prompt (intent JSON, plan JSON, answer text), so routing is testable.
- `embeddings.py`: `get_embeddings()` returns the local embedding model; the same model is used for indexing and querying.
- `tracing.py`: turns graph stream updates into `TraceEvent(node, started_at, ended_at, summary)` records.

## 6. Configuration and run modes

`.env.example` (committed) documents every variable; `.env` is gitignored.

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` or `fake` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | `http://ollama:11434` inside Compose, `http://host.docker.internal:11434` for a host Ollama |
| `OLLAMA_MODEL` | `qwen2.5:7b-instruct` *(decision 4, provisional)* | chat model tag |
| `OLLAMA_NUM_CTX` | `8192` | context window in tokens, 512–131072, sent as Ollama's `num_ctx` *(added)* |
| `OLLAMA_TIMEOUT_S` | `120.0` | HTTP timeout in seconds of each Ollama request, > 0 *(added)* |
| `LLM_TEMPERATURE` | `0.0` | sampling temperature, 0.0–2.0 *(added)* |
| `EMBEDDING_PROVIDER` | `huggingface` | `huggingface` or `fake` (offline hashing embeddings, no model download) *(added)* |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` *(decision 5, provisional)* | Hugging Face model id |
| `CHROMA_DIR` | `data/chroma_db` | index location |
| `CHROMA_COLLECTION` | `documents` | Chroma collection name *(added)* |
| `DATA_DIR` | `data/raw` | corpus location |
| `TOP_K` | `4` | retrieval depth |
| `MAX_RETRIES` | `2` | bound on the verify → re-plan loop |
| `INGEST_ON_START` | `true` | build the index if it is missing when the container starts (no effect until the Phase 6 entrypoint exists) |
| `LOG_LEVEL` | `INFO` | |

*(added)*: introduced while building the foundation or after its review. Every value is validated at start-up (for example `TOP_K` ≥ 1, `MAX_RETRIES` ≥ 0, `LLM_TEMPERATURE` between 0.0 and 2.0, and the project's rule for collection names: 3–63 characters, a deliberate limit stricter than chromadb's 512, and no IPv4 addresses); an empty value means the default, `.env` must be UTF-8, and `agentic-rag config` prints the effective values. The full reference is in [architecture.md](architecture.md#configuration-reference).

Run modes:

| Mode | LLM | Needs | Used for |
|---|---|---|---|
| `fake` | scripted fake | nothing | unit tests, CI, UI development, latency baseline without the LLM |
| `ollama-host` | Ollama on Windows (already installed) | `ollama serve` | fast local development |
| `ollama-compose` | Ollama container | Docker | the reproducible path reviewers run |

Fully offline fake mode also sets `EMBEDDING_PROVIDER=fake`; with the default `huggingface` provider the embedding model is downloaded on first use. In the `ollama-compose` mode, `compose.yaml` sets `OLLAMA_BASE_URL` for the `app` service itself.

## 7. Containerization plan

The `Dockerfile` bullet describes the image as built; the other deviations from this section are listed in [section 12.4](#124-containers-section-7).

- **`Dockerfile`** (required): two stages on `python:3.12.14-slim-trixie`, with `uv` mounted from `ghcr.io/astral-sh/uv:0.12.6` into the `RUN` steps only.
  - The `deps` stage runs `uv sync --locked --no-dev --no-install-project` into `/app/.venv`: only the dependencies pinned in `uv.lock`, and `--locked` stops the build when `uv.lock` is out of date with `pyproject.toml`.
  - The runtime stage copies the venv (`COPY --link`, 1.71 GB) and compiles its bytecode with `compileall` (415 MB) in two layers that do not depend on the code, creates the non-root user (`APP_UID`/`APP_GID` build arguments, default 10001), then adds `src/` and a small editable project layer with `uv sync --locked --no-dev --refresh-package agentic-rag-chatbot-poc`. A change to `src/` rebuilds only the last two layers (measured: 7 s; the image is 2.88 GB).
  - It sets `HF_HOME` to a volume-backed path, `EXPOSE 8501`, a healthcheck on `/_stcore/health`, and `CMD streamlit run src/agentic_rag/ui/app.py --server.address=0.0.0.0 --server.port=8501`. Pre-downloading the embedding model at build time for fully offline starts stays optional.
- **`.dockerignore`**: `.git`, `.venv`, `data/chroma_db`, `tests`, `docs`, `task`, `.claude`, caches, `.env`.
- **`compose.yaml`**:
  - `ollama`: `ollama/ollama`, named volume `ollama-data`, port `11434`, healthcheck via `ollama list`, GPU reservation under an optional `gpu` profile (Docker Desktop needs the WSL2 backend for NVIDIA pass-through; without it inference falls back to CPU).
  - `ollama-pull`: one-shot service that runs `ollama pull $OLLAMA_MODEL`, `depends_on: ollama` healthy.
  - `app`: `build: .`, port `8501`, `env_file: .env`, bind mount `./data:/app/data` so the corpus is visible and the index persists, named volume `hf-cache`, `depends_on: ollama-pull` completed successfully.
- Keep CPU-only torch wheels in the image (`[tool.uv.index]` pointing at the PyTorch CPU index) because the GPU is used by Ollama, not by the embedding model.
- Document in the README: `docker compose up --build`, the first-start duration (model pull + ingestion), and `LLM_PROVIDER=fake docker compose up app` for a model-free start.

## 8. Build order

Each phase ends in a committed state that passes its "done when" check. Phases 6–8 only need Phase 4; Phase 6 can start after Phase 1 in fake mode.

| Phase | Deliverables | Done when |
|---|---|---|
| **0. Lock decisions** | README *Design decisions* table filled for decisions 1–9; `docs/architecture.md` stub | Every row has a choice and a one-line trade-off |
| **1. Scaffold** (this plan's core) | `pyproject.toml`, `.python-version`, `uv.lock`, `src/agentic_rag/` with docstring-only modules, `config.py`, `cli.py` with stub commands, `.env.example`, `tests/conftest.py` + one smoke test, ruff config, `.dockerignore`, README *Repository structure* updated | `uv sync` succeeds; `uv run pytest` is green; `uv run python -m agentic_rag --help` lists the commands; `uv run ruff check .` is clean |
| **2. Ingestion and index** | corpus in `data/raw/` (or a download command), loaders, chunking, `index.py`, `ingest` command, fixture-corpus test | `python -m agentic_rag ingest` builds the index; a sample query returns the expected chunk with metadata |
| **3. RAG subgraph** | `rag/state.py`, `nodes.py`, `graph.py`, tests, Mermaid export | `rag_graph.invoke({"query": ...})` returns context and sources; tests pass in fake mode |
| **4. Main workflow and tools** | `agent/*`, fake LLM scripting, `export-graph` output replacing the hand-drawn diagrams in `docs/architecture.md` | ≥ 5 nodes, routing tests cover all four intents, decomposition runs via `Send`, the retry loop is bounded, an end-to-end answer works with Ollama |
| **5. Streamlit UI** | `ui/app.py`, `components.py`, `AppTest` smoke test | Chat answers, steps appear live, sources are shown, runs with `LLM_PROVIDER=fake` |
| **6. Containerization** | `Dockerfile`, `compose.yaml`, entrypoint with optional ingest, README run guide | From a fresh clone `docker compose up --build` serves the UI on 8501 with the model pulled and the index built; `docker build .` alone succeeds |
| **7. Functional evaluation** | `data/eval/questions.jsonl` (10–20), `evaluation/*`, `eval` command, `docs/evaluation.md`, README section | `python -m agentic_rag eval` writes the results and the summary; conclusions are in the README |
| **8. Load test** | `loadtest/runner.py`, `loadtest` command, `docs/performance.md`, README section | `python -m agentic_rag loadtest --requests 100 --concurrency 4` prints percentiles and the per-node breakdown; bottleneck and 1–2 proposals documented |
| **9. Docs and polish** | README EN + HU complete, requirement checklist ticked, optional CI workflow, final clean-clone test | A reviewer can reproduce every claim with the documented commands |

Progress on 2026-10-01:

- **Phase 0** is partly done: decisions 1–7 are in the README *Design decisions* table (4 and 5 with provisional defaults), decisions 8 and 9 are still open, and the `docs/architecture.md` stub exists.
- **Phase 1** is done: its four checks pass, and `--help` lists `ingest`, `eval`, `loadtest`, `export-graph` and `config`.
- **Brought forward** into the foundation, built and tested ahead of their phases:
  - from Phase 6: the `Dockerfile`, `compose.yaml`, the GPU override `compose.gpu.yaml` and the README run guide. The image builds and the `app` service runs healthy in fake mode. Still in Phase 6: the entrypoint with the optional ingestion (`INGEST_ON_START`) and the fresh-clone run of the full stack, with the model pulled and the index built;
  - the shared modules of section 5.7: `config.py`, `llm.py` (`ChatOllama` and the scripted fake model; the fake's rules for the real prompts stay in Phase 4), `embeddings.py` (sentence-transformers and an offline fake) and `tracing.py`;
  - from Phases 3–4: the state contracts (`rag/state.py`, `agent/state.py`) and the interface of the `search_knowledge_base` tool;
  - from Phase 5: the Streamlit shell (`ui/app.py`, `ui/components.py`) with its `AppTest` tests in `tests/test_ui.py`; still in Phase 5: the check against the real graph;
  - from Phases 2, 7 and 8: the data contracts and pure helpers they build on: the document and chunk metadata, the chunking defaults and `IndexStats`; the question-set loader, hit@k, routing accuracy and the report models; the latency statistics.
- **Skeletons:** every other public function of Phases 2–8 exists as a typed stub and raises `agentic_rag.errors.PlannedFeatureError` (a `NotImplementedError` subclass) through `planned(...)`, with the message `<qualified name> is planned for Phase <N> (see docs/project-structure-plan.md, section 8)`. These messages and the tests cite this section, so keep its number and the phase numbers stable.
  - **Fixed now:** the names (modules, public functions and classes, `NODE_NAMES`, `RAG_NODE_NAMES`, `NODE_TARGETS`); the dependency convention (the state is a node's only positional parameter, its dependencies are keyword-only and bound by the graph builder, and every library factory takes `settings` as a required argument; see [dependency injection and cheap builds](architecture.md#dependency-injection-and-cheap-builds)); and the data contracts (the state schemas and records, the report models, the question-set format).
  - **May still be extended:** a later phase may add keyword-only parameters, for example a node dependency found while writing its prompt, or narrow a return type, as long as existing callers and the tests of these contracts keep working. Such changes are recorded in section 12.
- **Phase 4 note:** the `export-graph` command already exists (`--graph`, `--format`, `--output`) and draws the graphs once they are built. It writes a complete Markdown file, so its output is pasted into `docs/architecture.md`, or written to a file of its own, rather than pointing `--output` at the hand-written document.

## 9. Requirement traceability

| Requirement (brief) | Where | Verified by |
|---|---|---|
| Real problem with justification | README *Problem statement*, decision 8 | Review |
| Freely chosen text source, quality processing | `data/raw/`, `ingestion/` | `test_ingestion.py`, `ingest` command |
| ≥ 5 nodes | `agent/graph.py` | `test_agent_graph.py` asserts the node count |
| Conditional routing | `agent/routing.py` | Routing tests per intent |
| Decomposition and independent execution | `plan_subtasks` + `Send` fan-out | Test with a multi-part question |
| State for intermediate results | `agent/state.py` | Reducer tests |
| ≥ 2 tools, one non-retrieval | `agent/tools.py` | `test_tools.py` |
| Modular RAG subgraph, callable, not counted | `rag/graph.py` | `test_rag_subgraph.py`, Mermaid export with `xray=True` |
| Open-source local LLM, justified; dummy allowed | `llm.py`, decisions 3–4, README | Runs with both providers |
| Streamlit UI with steps and RAG result | `ui/` | `test_ui_smoke.py`, manual check |
| Dockerfile, Compose | `Dockerfile`, `compose.yaml` | Clean-clone `docker compose up --build` |
| Evaluation set of 10–20 questions | `data/eval/questions.jsonl`, `evaluation/` | `eval` command, `docs/evaluation.md` |
| Load test 50–200 queries, latency, bottleneck, proposals | `loadtest/`, `docs/performance.md` | `loadtest` command |
| README: problem, architecture, results, run guide | `README.md`, `README.hu.md` | Checklist ticked |

## 10. Conventions

- Code, identifiers, comments and commit messages in English; user-facing documentation in English and Hungarian.
- `src` layout, type hints everywhere, module docstrings, `ruff` for lint and format, `pytest` for tests.
- Every test passes with `LLM_PROVIDER=fake` and without network access; model-dependent checks are marked and skipped when Ollama is unavailable.
- Nodes return partial state updates and never mutate state; every accumulated list has a reducer; loops go through named nodes with a bounded counter.
- Configuration only through `Settings`; no secrets in code, images or Compose files.
- Diagrams are generated from the compiled graphs (`draw_mermaid()`), not drawn by hand, so `docs/architecture.md` cannot drift from the code.
- Every documented number (evaluation score, latency) has a command that reproduces it and a committed result file.
- README *Repository structure* and the requirement checklist are updated at the end of each phase.

## 11. Risks and open questions

- **Image size**: torch plus `sentence-transformers` adds ~1 GB even with CPU wheels. Fallback: Ollama embeddings (`nomic-embed-text`, `bge-m3`) and a torch-free image.
- **GPU in Docker on Windows**: requires the WSL2 backend and a current NVIDIA driver. Fallback: host Ollama via `host.docker.internal`, or CPU inference with a 3B model.
- **Cold start**: the first Ollama request loads the model (seconds); the load test must include a warm-up phase and report it separately.
- **Hungarian quality of small models**: if the corpus and questions are Hungarian, prefer a model with explicit multilingual coverage and verify it on the evaluation set before committing to it.
- **Corpus licensing**: only commit documents whose license allows redistribution; otherwise ship a download command and record the source URLs.
- **Open**: the final domain and corpus (decision 8), the non-retrieval tool (decision 9), the UI language, and whether a FastAPI service is worth adding as a third Compose component for a more realistic load test.

## 12. Deviations from the plan

Recorded while building the foundation and after its review (2026-10-01). The sections above keep the original plan, except for the settings table of section 6 and the `Dockerfile` bullet of section 7, which describe the code; where they differ, the code and this list are current. The cross-cutting contracts are described once, in [architecture.md](architecture.md).

### 12.1 Configuration (section 6)

- Five settings were added: `LLM_TEMPERATURE`, `EMBEDDING_PROVIDER` (`huggingface` or `fake`, so tests and model-free demos need no embedding download), `CHROMA_COLLECTION`, and, after the review, `OLLAMA_NUM_CTX` (default 8192, 512–131072) and `OLLAMA_TIMEOUT_S` (default 120.0, > 0). The last two reach `ChatOllama` as `num_ctx` and as the timeout of its HTTP clients: without `num_ctx` the server's small default context length would silently truncate the prompts, and the HTTP timeout is the only bound on a slow request, because LangGraph cannot time out a sync node.
- `CHROMA_COLLECTION` follows the project's own rule: 3–63 characters from `[A-Za-z0-9._-]`, a letter or digit at both ends, no `..`, and not an IPv4 address. The 63-character limit is deliberately stricter than chromadb 1.5.9 (3–512), to keep names portable; the IPv4 rule matches chromadb's own check. A test checks the boundary names against the installed chromadb.
- `get_settings()` raises `agentic_rag.errors.ConfigurationError` for a `.env` that cannot be read or is not UTF-8 (Windows PowerShell 5.1 writes UTF-16 with `>`), and `config.describe_invalid_settings()` names the invalid variables for the CLI and the UI alike.
- `INGEST_ON_START` exists but has no effect yet: nothing ingests at start-up until the Phase 6 entrypoint, which needs Phase 2's `build_index`. The UI does not ingest, to keep its start-up light.
- Chunk size and overlap are code constants (`ChunkingConfig`: 900 characters with a 150-character overlap, inside the range of section 5.4), not settings.

### 12.2 Shared modules and contracts (sections 5.1–5.3 and 5.7)

- Trace events are recorded by the `@traced` node decorator, which appends them to each node's state update, instead of being derived from the stream. They therefore travel in the state: an `invoke` result carries the whole trace, including the RAG subgraph's events, which `run_rag_subtask` forwards. `TraceEvent` also has `duration_ms` and `metadata`.
- `search_knowledge_base` is a `BaseTool` subclass that holds the compiled RAG subgraph, with `response_format="content_and_artifact"` (the context for a model, the whole `RagOutput` for the application), rather than an `@tool` function, because it needs a dependency built at graph-build time. `run_rag_subtask` calls the subgraph through this tool, so `get_graph(xray=True)` does not nest the subgraph in the main diagram: `export-graph` draws it as a diagram of its own, which is also what the Mermaid export cited in section 9 shows.
- `analyze_request` also writes `draft_answer` on the `direct` route and a one-step `subtasks` plan on the `single` and `tool` routes. `plan_subtasks` resets `subtask_results` with `Overwrite` at the start of every planning round, so a re-plan does not mix in the rejected round's results.
- The states are `TypedDict`s with `total=False` and explicit input and output schemas (`AgentInput` and `AgentOutput`, `RagInput` and `RagOutput`); `RagState` also has a `trace` key.
- The fake LLM is a rule engine (`ScriptedChatModel`: ordered regular expressions, JSON structured output); its rules for the real prompts come in Phase 4. `EMBEDDING_PROVIDER=fake` adds an offline embedding fake (a hashed bag of words) next to the planned sentence-transformers model.
- The evaluation item also has `id`, `tags` and `notes`, and unknown keys are rejected. The load-test statistics also report the minimum, and the warm-up requests are summarized separately.

Changed after the review:

- **Trace contract trimmed.** `@traced` accepts a `dict` or `None` result only and raises `TypeError` otherwise, `Command` included; `trace_events_from_chunk` accepts `version="v2"` stream parts, plain `{node: update}` mappings and `(namespace, mapping)` tuples. Traced nodes get no LangGraph `CachePolicy`, because a cache hit would replay the old `TraceEvent`; caching happens inside a node. `tracing.py` imports no LangGraph.
- **Node convention for both graphs.** The RAG nodes now take keyword-only dependencies like the agent nodes: `rewrite_query(state, *, chat_model)`, `retrieve(state, *, vector_store, top_k)`, `grade_documents(state, *, min_score, chat_model)` and `build_context(state)`; `build_rag_graph(settings)` binds them with `functools.partial`, the vector store through a lock-guarded lazy provider of `load_index(settings)`. Library factories take `settings` as a required argument, without a `get_settings()` fallback.
- **Execution model.** The nodes are sync. Instead of the async harness of section 5.6, the load test calls `graph.invoke` from a `ThreadPoolExecutor(max_workers=concurrency)`; the async overrides of the fakes and the `_arun` hint of the search tool were removed.
- **Citations.** `build_context` numbers its markers per RAG run in rank order (`sources[i]` is `[i + 1]`), and `synthesize_answer` assigns one global numbering across the sub-tasks, which `finalize_response` returns as `sources`.
- **Errors.** `agentic_rag.errors` defines `PlannedFeatureError` (a `NotImplementedError` subclass that every stub raises through `planned()`), `ConfigurationError` and `InvalidArgumentError`. The CLI and the UI treat only `PlannedFeatureError` as a planned gap; the CLI maps `InvalidArgumentError`, `ConfigurationError` and invalid settings to exit code 2.
- **Shared modules.** `agent/types.py` holds `Intent`, `Verdict` and `SubtaskKind` without LangGraph imports (`agent/state.py` re-exports them), and `reports.py` holds `RESULTS_DIR` and `RunReport`, the base of `EvalReport` and `LoadTestReport`, so loading a report loads no LangGraph. The checkpointer allowlist `STATE_RECORD_TYPES` was removed; the no-checkpointer note in `agent/state.py` says what a checkpointer would need.
- **Evaluation.** `expected_sources` became `expected_documents` (paths relative to `DATA_DIR` with forward slashes, or URLs) and `retrieved_sources` became `retrieved_documents`, one ranked list per retrieve sub-task; hit@k is computed per retrieve sub-task from `SubtaskResult.sources`, never from `AgentOutput.sources`. `eval --target node` accepts only `NODE_TARGETS` (`analyze_request`, `run_rag_subtask`); `run_evaluation` raises `InvalidArgumentError` for any other node. `EvalItemResult` rejects verdicts that contradict its own item.
- **Ingestion.** `build_index` reconciles the collection with the corpus by set difference: it upserts every produced chunk, then deletes every stored id the run did not produce, so a plain `ingest` handles edited files too and `--rebuild` is needed only after changing the embeddings. Support for the instruct prefixes of E5 models was dropped; the `query:` / `passage:` prefixes stay.

### 12.3 UI (section 5.5)

- The UI streams with `stream_mode=["updates", "values"]` (`version="v2"`) instead of `"updates"` alone: the updates feed the step panel, and the last root values give the answer and the sources. The step panel lists the main graph's own steps; the RAG subgraph's inner steps are not listed, and its result appears in the retrieved-context panel.
- The sidebar shows the provider, the models and top-k read-only. They are changed through environment variables or `.env` and a restart, not through widgets.
- Added after the review: a run the user stops gets the turn *Stopped before an answer was produced.*, so the history never keeps an unanswered question; the agent receives the new question with only the earlier questions that were answered and their answers; `$` signs outside code are escaped when an answer is rendered, so amounts do not turn into LaTeX; only `PlannedFeatureError` is shown as a notice, every other exception is logged and shown with `st.exception`; invalid settings or an unreadable `.env` replace the chat with an *Invalid configuration* error.
- The step panel groups parallel steps by overlapping time windows, so `Send` workers that do not overlap in time show as sequential steps; grouping by LangGraph step is left to Phase 5.

### 12.4 Containers (section 7)

- `.dockerignore` is an allowlist: everything is excluded, then `pyproject.toml`, `uv.lock`, `.python-version`, `README.md`, `LICENSE` and `src/` are re-included.
- The images are pinned to exact tags: `python:3.12.14-slim-trixie`, `ghcr.io/astral-sh/uv:0.12.6` and `ollama/ollama:0.35.0`. There is no entrypoint script yet, only `CMD`.
- The Ollama port is not published on the host, because the host may already run Ollama on 11434 and the API has no authentication; a local, gitignored `compose.override.yaml` can publish it.
- GPU support is an override file, `compose.gpu.yaml`, instead of a `gpu` profile: a profile switches whole services, so it would need a second `ollama` service, and `depends_on` cannot point at either of two services.
- The `app` service mounts `./data/raw` read-only instead of `./data`, and the index lives in the named volume `chroma-data`. `eval` and `loadtest` are therefore recommended on the host. In the container they need an extra `./data/eval` bind mount, which the app user (UID/GID 10001) must be able to write: on a Linux engine a bind mount keeps the host owner, so either make `data/eval/results` writable for UID 10001 or build with `APP_UID=$(id -u) APP_GID=$(id -g) docker compose build app`.
- After the review: the `Dockerfile` uses `uv sync --locked` instead of `--frozen`, so a stale `uv.lock` stops the build, and is split into a `deps` stage and a runtime with separate venv, bytecode, `src/` and project layers (section 7), so code changes no longer rebuild the 2 GB dependency layer. `compose.yaml` passes `APP_UID` and `APP_GID` (default 10001) as build arguments. The `docker run` example uses `--mount type=bind,source=./data/raw,target=/app/data/raw,readonly`, which Git Bash does not rewrite.
- `.gitattributes` keeps LF line endings in every checkout (`* text=auto eol=lf`, CRLF only for `*.bat` and `*.cmd`), so Windows clones with `core.autocrlf=true` build the same image and a future shell entrypoint keeps working. `.gitignore` anchors the directory patterns that are also common package names (`/build/`, `/dist/`, `/env/`, `/venv/`, `/models/`, `/task/`) to the repository root.
- `.env` is optional for Compose (`required: false`). `compose.yaml` passes `LLM_PROVIDER`, `EMBEDDING_PROVIDER` and `OLLAMA_MODEL` from the shell or `.env`, and fixes `OLLAMA_BASE_URL` to the `ollama` service.
- `ollama-pull` runs `ollama show … || ollama pull …`, so it downloads only a missing model, and `app` also waits for `ollama` to be healthy.
- The model-free start is `LLM_PROVIDER=fake EMBEDDING_PROVIDER=fake docker compose up --build --no-deps app`: without `--no-deps` Compose would also start the Ollama services, and without `EMBEDDING_PROVIDER=fake` the embedding model would be downloaded.
- The embedding model is not pre-downloaded at build time; this stays optional.

### 12.5 Tooling and documentation

- Ruff skips `.claude/` (third-party agent and skill files) and does not format `docs/*.md`, whose code snippets are illustrative.
- The foundation's tests are named after the modules they cover (`test_config.py` … `test_ui.py`). The test files of section 4 (`test_ingestion.py`, `test_rag_subgraph.py`, `test_agent_graph.py`, `test_tools.py`) come with their phases; the `AppTest` tests of the UI are in `tests/test_ui.py` rather than `test_ui_smoke.py`.
- Section 10 asks for generated diagrams. Until Phases 3–4 build the graphs, `docs/architecture.md` holds hand-drawn target diagrams, labelled as such.
- Section 10 says model-dependent checks are skipped when Ollama is unavailable. The live Ollama test is instead deselected by default (`addopts = -m "not ollama"`), so `uv run pytest` reports `1 deselected` and never calls a model; `uv run pytest -m ollama` runs it and skips it when the server is unreachable.
- Ruff's pydocstyle rules `D1` (google convention) require a docstring on every public module, class and function in `src/`; `tests/` are exempt. Docstrings mark code with double backticks.
- The cross-cutting contracts now have one canonical description in [architecture.md](architecture.md), while the module docstrings still restate parts of them. Consolidating the docstrings so that each contract is stated once is planned together with Phase 4, when the node bodies replace most of the stub docstrings.
