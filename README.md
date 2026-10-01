# Agentic RAG Chatbot – Proof of Concept

**English** | [Magyar](README.hu.md)

An agentic Retrieval-Augmented Generation (RAG) chatbot prototype in Python — built with [LangGraph](https://github.com/langchain-ai/langgraph), powered by a locally hosted open-source LLM, with a [Streamlit](https://streamlit.io/) UI, and fully containerized with Docker.

> **Status:** 🚧 Work in progress. This README already covers the project's goals, requirements and documentation structure; sections marked *To be completed* will be filled in as the implementation progresses.

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

This project is a solution to a technical assignment for a Medior AI Engineer position; the original brief (in Hungarian) is in the `task/` folder.

## Assignment requirements

Status of each requirement from the brief:

**Problem & data**

- [ ] Real-world problem (domain / use case) with a written justification
- [ ] Freely chosen text data source, with the focus on quality processing and scalable data integration rather than volume

**Agentic architecture (LangGraph)**

- [ ] Agentic workflow with at least 5 nodes
- [ ] Autonomous decision-making (e.g. conditional routing)
- [ ] Decomposition into sub-tasks and their independent execution
- [ ] State management for storing intermediate results
- [ ] At least 2 tools, at least one of which is not purely retrieval-based
- [ ] Dedicated, modular RAG subgraph callable from the main workflow (not counted towards the node count)

**Model, UI & deployment**

- [ ] Open-source LLM that fits the local resources (no paid APIs), with the trade-offs justified
- [ ] Streamlit prototype UI that shows the agent's main steps and the result of the RAG process
- [ ] Containerized: `Dockerfile` (required) and `docker-compose.yml` (a plus for multi-component setups)

**Evaluation & performance**

- [ ] Functional evaluation on a mini set of 10–20 questions (a single node or the full workflow)
- [ ] Load test with 50–200 queries: basic latency metrics, the main bottleneck, 1–2 concrete optimization proposals

**Documentation**

- [ ] This README: problem & goals, architecture & design rationale, evaluation & load-test results, setup & run guide

## Problem statement and motivation

> 🚧 *To be completed* — the chosen domain / use case and the goal of the chatbot, answering three questions:
>
> - **Why is the problem relevant?**
> - **What user need does it address?**
> - **Why is an agentic RAG approach a good fit** — compared to a single retrieve-then-generate pass?

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

> 🚧 *To be completed:* the main workflow's nodes and routing logic, the RAG subgraph's steps, the tools, the state schema and the ingestion pipeline.
>
> Tip: LangGraph can export the compiled graph as Mermaid with `graph.get_graph(xray=True).draw_mermaid()` — `xray=True` also expands the RAG subgraph.

## Design decisions

| Area | Key trade-offs | Choice & rationale |
|---|---|---|
| Domain & data source | Relevance, availability and licensing, preprocessing effort | _TBD_ |
| LLM | Answer quality vs. latency vs. memory (RAM/VRAM); tool-calling support; license | _TBD_ |
| LLM serving | Setup effort, containerization, throughput | _TBD_ |
| Embedding model | Retrieval quality vs. speed; language coverage | _TBD_ |
| Vector store | Persistence, metadata filtering, scalability | _TBD_ |
| Chunking | Chunk size and overlap vs. retrieval precision and context length | _TBD_ |

## Evaluation

### Functional evaluation

**Approach:** a mini evaluation set of 10–20 domain questions — each with a reference answer and, where relevant, the expected source documents — used to evaluate a single node or the full agentic workflow.

**Candidate metrics:** answer correctness against the reference, faithfulness to the retrieved context, retrieval hit rate@k, and routing / tool-selection accuracy.

> 🚧 *To be completed:* location of the evaluation set, scoring method, results, conclusions and the command to reproduce them.

### Load test & bottleneck analysis

**Scenario:** 50–200 queries against the running system, with the concurrency level, query mix and hardware documented.

**Reported metrics:** latency (mean, p50, p95, p99, max), throughput and error rate, plus a per-node latency breakdown to pinpoint the main bottleneck — followed by 1–2 concrete optimization proposals.

> 🚧 *To be completed:* results, bottleneck analysis, optimization proposals and the command to reproduce them.

## Getting started

### Prerequisites

- Git
- Docker with Docker Compose v2
- Enough memory to run the chosen LLM locally (_exact RAM/VRAM requirements TBD_)

### Run with Docker Compose

> The `Dockerfile` and `docker-compose.yml` are not in the repository yet — the commands below show the intended workflow.

```bash
git clone https://github.com/Csabikahhh/agentic-rag-chatbot-poc.git
cd agentic-rag-chatbot-poc
docker compose up --build
```

Then open the UI at <http://localhost:8501>.

> 🚧 *To be completed:* environment variables, model download, data ingestion, running locally without Docker, and the commands for the evaluation and the load test.

## Repository structure

```text
agentic-rag-chatbot-poc/
├── docs/          # Supplementary documentation (diagrams, evaluation and performance reports)
├── task/          # Original assignment brief (Hungarian)
├── LICENSE
├── README.md      # Documentation (English)
└── README.hu.md   # Documentation (Hungarian)
```

The structure will grow with the application code, the container setup, the evaluation set and the load-test scripts.

## License

Released under the [MIT License](LICENSE). © 2026 Csaba Ovari
