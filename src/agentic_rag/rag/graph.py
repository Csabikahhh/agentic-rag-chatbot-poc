"""Wiring of the RAG subgraph (plan section 5.2): the compiled graph the main workflow calls.

:data:`RAG_NODE_NAMES` fixes the node names and their order. :func:`build_rag_graph` is
planned for Phase 3:

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
- ``rewrite_query`` and ``grade_documents``: ``chat_model`` is
  ``agentic_rag.llm.get_chat_model(settings)`` with the ``ollama`` provider. With the
  ``fake`` provider it is None, which skips the rewrite (plan section 5.2 skips it in fake
  mode) and the optional LLM grade. Creating ``ChatOllama`` makes no connection.
- ``retrieve``: ``top_k=settings.top_k``, and ``vector_store``, a provider that the builder
  creates for this graph. On its first call the provider runs ``load_index(settings)``
  under a ``threading.Lock``; later calls return the same store. Parallel ``Send`` workers of
  the main graph therefore share one index and one embedding model. A bare
  ``functools.cache`` is not enough: it lets concurrent first calls each run ``load_index``.
- ``grade_documents``: ``min_score``, the relevance threshold. Phase 3 calibrates it on the
  evaluation set for each embedding provider, because the score ranges of the E5 model and
  of the hashing fake differ.

The nodes read their configuration only from these bindings, never from ``get_settings()``,
so the settings passed to the builder decide everything. Building stays cheap: no model is
loaded and the index is not opened while building, so ``agentic-rag export-graph`` and the UI
start-up can compile and draw the graph quickly.
"""

from typing import Final

from langgraph.graph.state import CompiledStateGraph

from agentic_rag.config import Settings
from agentic_rag.errors import planned
from agentic_rag.rag.state import RagInput, RagOutput, RagState

__all__ = ["RAG_NODE_NAMES", "build_rag_graph"]

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


def build_rag_graph(settings: Settings) -> CompiledStateGraph[RagState, None, RagInput, RagOutput]:
    """Build and compile the RAG subgraph.

    Args:
        settings: The settings the nodes are bound to (see the module docstring):
            ``top_k``, the LLM and embedding providers, and the index location.

    Returns:
        The compiled subgraph. It starts from a ``RagInput`` and ``invoke({"query": ...})``
        returns a ``RagOutput``. The first run opens the index; building does not.

    Raises:
        PlannedFeatureError: Until Phase 3 implements the function.
    """
    raise planned(f"{__name__}.build_rag_graph", 3)
