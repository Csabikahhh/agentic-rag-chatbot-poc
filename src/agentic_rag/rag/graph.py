"""Wiring of the RAG subgraph (plan section 5.2): the compiled graph the main workflow calls.

:data:`RAG_NODE_NAMES` fixes the node names and their order. :func:`build_rag_graph`:

- ``StateGraph(RagState, input_schema=RagInput, output_schema=RagOutput)``: callers pass
  ``{"query": ...}`` and get exactly the ``RagOutput`` keys back.
- The nodes of ``agentic_rag.rag.nodes`` run linearly: START -> ``rewrite_query`` ->
  ``retrieve`` -> ``grade_documents`` -> ``build_context`` -> END. There is no conditional
  edge: retrying with a better query is the main workflow's job (``verify_answer`` ->
  ``plan_subtasks``), and the linear path guarantees that ``build_context`` always runs.
- It is compiled without a checkpointer. The main workflow invokes it from its
  ``run_rag_subtask`` node with an explicit input/output mapping (plan section 5.1); tests,
  the evaluation and the load test invoke it on its own. Its four nodes do not count towards
  the main workflow's minimum of five nodes.

How ``build_rag_graph(settings)`` binds the nodes' keyword-only dependencies:

- Each node is registered under its explicit name, ``add_node(name, functools.partial(node,
  ...))``, because a partial has no ``__name__``. The partial also hides the state type hint,
  so LangGraph falls back to the graph's state schema, ``RagState``, which is what the nodes
  read.
- ``rewrite_query``: ``chat_model`` is ``agentic_rag.llm.get_chat_model(settings)`` with the
  ``ollama`` provider. With the ``fake`` provider it is None, which skips the rewrite (plan
  section 5.2 skips it in fake mode). Creating ``ChatOllama`` makes no connection.
- ``grade_documents``: the same chat model when ``settings.grade_with_llm`` is on, None
  otherwise and with the ``fake`` provider, which skips the LLM grade. ``min_score`` is
  :data:`MIN_SCORES` of the embedding provider.
- ``retrieve``: ``top_k=settings.top_k``, and ``vector_store``, a provider that the builder
  creates for this graph. On its first call the provider runs ``load_index(settings)``
  under a ``threading.Lock``; later calls return the same store. Parallel ``Send`` workers of
  the main graph therefore share one index and one embedding model. A bare
  ``functools.cache`` is not enough: it lets concurrent first calls each run ``load_index``.

The nodes read their configuration only from these bindings, never from ``get_settings()``,
so the settings passed to the builder decide everything. Building stays cheap: no model is
loaded and the index is not opened while building, so ``agentic-rag export-graph`` and the UI
start-up can compile and draw the graph quickly.
"""

import functools
import itertools
import threading
from collections.abc import Callable
from typing import Final

from langchain_core.vectorstores import VectorStore
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agentic_rag.config import EmbeddingProvider, Settings
from agentic_rag.ingestion.index import load_index
from agentic_rag.llm import get_chat_model
from agentic_rag.rag import nodes
from agentic_rag.rag.state import RagInput, RagOutput, RagState

__all__ = ["MIN_SCORES", "RAG_NODE_NAMES", "build_rag_graph"]

RAG_NODE_NAMES: Final[tuple[str, ...]] = (
    "rewrite_query",
    "retrieve",
    "grade_documents",
    "build_context",
)
"""Names of the RAG subgraph nodes in execution order (plan section 5.2).

Each name is also the name of the node function in ``agentic_rag.rag.nodes`` and the
``node`` of the ``TraceEvent`` it records.
"""

MIN_SCORES: Final[dict[EmbeddingProvider, float]] = {"huggingface": 0.83, "fake": 0.0}
"""Relevance threshold of ``grade_documents`` for each embedding provider; provisional.

The score ranges differ, so each provider has its own value. Measured on the built index
(2026-10-02) with eight frontend questions and eight unrelated ones:

- ``huggingface`` (``intfloat/multilingual-e5-small``): the best chunk of a frontend question
  scored 0.89-0.93, of an unrelated question 0.72-0.86. 0.83 drops what is clearly unrelated
  and leaves the borderline cases to the LLM grade.
- ``fake`` (hashed bag of words): the two groups overlap (0.28-0.59 against 0.27-0.43), so no
  threshold separates them; 0.0 keeps every chunk.

The evaluation set (Phase 7) measures the values again.
"""


def build_rag_graph(settings: Settings) -> CompiledStateGraph[RagState, None, RagInput, RagOutput]:
    """Build and compile the RAG subgraph.

    Args:
        settings: The settings the nodes are bound to (see the module docstring):
            ``top_k``, ``grade_with_llm``, the LLM and embedding providers, and the index
            location.

    Returns:
        The compiled subgraph. It starts from a ``RagInput`` and ``invoke({"query": ...})``
        returns a ``RagOutput``. The first run opens the index; building does not.
    """
    chat_model = get_chat_model(settings) if settings.llm_provider != "fake" else None
    grading_model = chat_model if settings.grade_with_llm else None

    builder = StateGraph(RagState, input_schema=RagInput, output_schema=RagOutput)
    builder.add_node("rewrite_query", functools.partial(nodes.rewrite_query, chat_model=chat_model))
    builder.add_node(
        "retrieve",
        functools.partial(
            nodes.retrieve, vector_store=_index_provider(settings), top_k=settings.top_k
        ),
    )
    builder.add_node(
        "grade_documents",
        functools.partial(
            nodes.grade_documents,
            min_score=MIN_SCORES[settings.embedding_provider],
            chat_model=grading_model,
        ),
    )
    builder.add_node("build_context", nodes.build_context)

    builder.add_edge(START, RAG_NODE_NAMES[0])
    for current, following in itertools.pairwise(RAG_NODE_NAMES):
        builder.add_edge(current, following)
    builder.add_edge(RAG_NODE_NAMES[-1], END)
    return builder.compile(name="rag_subgraph")


def _index_provider(settings: Settings) -> Callable[[], VectorStore]:
    """Return a provider that opens the index once, on its first call, safely across threads."""
    lock = threading.Lock()
    opened: list[VectorStore] = []

    def provide() -> VectorStore:
        if not opened:
            with lock:
                if not opened:
                    opened.append(load_index(settings))
        return opened[0]

    return provide
