"""Node functions of the RAG subgraph (plan section 5.2), planned for Phase 3.

The four nodes run in this order and exchange data only through ``RagState``:

- ``rewrite_query`` reads ``query`` and writes ``rewritten_query``.
- ``retrieve`` reads ``rewritten_query`` (or ``query``) and writes ``documents`` and
  ``scores``.
- ``grade_documents`` reads ``documents`` and ``scores`` and writes the relevant subset of
  both.
- ``build_context`` reads ``documents`` and ``scores`` and writes ``context`` and ``sources``.

Contract for the implementation:

- The node convention of both graphs, described in ``agentic_rag.agent.graph``, applies.
  The state is the only positional parameter. Dependencies are keyword-only parameters
  without defaults, which ``agentic_rag.rag.graph.build_rag_graph`` binds once per compiled
  graph, so tests can call a node directly with stand-ins. Each node takes only what its role
  in plan section 5.2 needs:

  - ``rewrite_query`` (optional LLM rewrite): ``chat_model``, or None to skip the rewrite.
  - ``retrieve`` (top-k similarity search): ``vector_store``, a zero-argument provider of the
    index, and ``top_k``.
  - ``grade_documents`` (score threshold, optionally an LLM grade): ``min_score``, and
    ``chat_model``, or None to skip the LLM grade.
  - ``build_context`` (de-duplicate, order, format): nothing; it only formats.

- Nodes are synchronous functions. They return partial updates and never mutate the state.
- ``documents`` and ``scores`` are parallel lists (``scores[i]`` belongs to ``documents[i]``).
  Higher scores are more relevant, and the lists stay in rank order, most relevant first,
  from ``retrieve`` to ``build_context``.
- Every node is wrapped with ``agentic_rag.tracing.traced`` and its function name equals its
  node name in ``agentic_rag.rag.graph.RAG_NODE_NAMES``. So each executed node appends one
  ``TraceEvent`` to ``trace``, and ``trace_events_from_chunk(chunk, skip_forwarded=True)``
  can tell a node's own events from forwarded ones.
- ``build_context`` runs on every path, also when nothing relevant was found (``context`` is
  then empty and ``sources`` is ``[]``), because ``RagOutput`` is total.
- ``Source`` fields come from ``ChunkMetadata``; ``page`` is already 1-based there.
"""

from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

from agentic_rag.errors import planned
from agentic_rag.rag.state import RagState
from agentic_rag.tracing import traced

__all__ = ["build_context", "grade_documents", "retrieve", "rewrite_query"]


@traced
def rewrite_query(state: RagState, *, chat_model: BaseChatModel | None) -> dict[str, Any]:
    """Rewrite the query into a standalone search query.

    Reads ``query``. Writes ``rewritten_query``: a short, self-contained search query
    written by ``chat_model``, or ``query`` unchanged when ``chat_model`` is None.

    Args:
        state: The subgraph state.
        chat_model: The chat model that rewrites the query, or None to skip the rewrite.
            ``build_rag_graph`` passes None with the fake LLM provider, because plan section
            5.2 skips the rewrite in fake mode.

    Returns:
        The partial update ``{"rewritten_query": ...}``.

    Raises:
        PlannedFeatureError: Until Phase 3 implements the node.
    """
    raise planned(f"{__name__}.rewrite_query", 3)


@traced
def retrieve(
    state: RagState, *, vector_store: Callable[[], VectorStore], top_k: int
) -> dict[str, Any]:
    """Search the vector index for the chunks closest to the query.

    Reads ``rewritten_query``, or ``query`` when no rewrite was written. Writes
    ``documents``, the ``top_k`` most similar chunks with their ``ChunkMetadata``, most
    relevant first, and ``scores``, their relevance scores in the same order.

    Args:
        state: The subgraph state.
        vector_store: Returns the vector index of ``agentic_rag.ingestion.index.load_index``.
            The node calls it on every run. The provider opens the index on its first call
            and returns the same store afterwards, so the first query pays for loading the
            embedding model, not the graph build.
        top_k: Number of chunks to retrieve (``Settings.top_k``).

    Returns:
        The partial update ``{"documents": [...], "scores": [...]}``.

    Raises:
        IndexNotFoundError: From ``vector_store``, if the index has not been built yet.
        EmbeddingMismatchError: From ``vector_store``, if the index was built with another
            embedding provider or model.
        PlannedFeatureError: Until Phase 3 implements the node.
    """
    raise planned(f"{__name__}.retrieve", 3)


@traced
def grade_documents(
    state: RagState, *, min_score: float, chat_model: BaseChatModel | None
) -> dict[str, Any]:
    """Drop the retrieved chunks that are not relevant to the query.

    Reads ``documents`` and ``scores``, and ``rewritten_query`` or ``query`` for the LLM
    grade. Writes ``documents`` and ``scores`` again with the relevant subset in rank order.
    Chunks that score below ``min_score`` are dropped first. Then, if ``chat_model`` is set,
    the chunks it grades as irrelevant are dropped too (plan section 5.2). Both lists may
    become empty.

    Args:
        state: The subgraph state.
        min_score: The lowest relevance score a chunk may have and still be kept.
        chat_model: The chat model that grades the chunks above the threshold, or None to
            keep all of them. ``build_rag_graph`` passes None with the fake LLM provider.

    Returns:
        The partial update ``{"documents": [...], "scores": [...]}``.

    Raises:
        PlannedFeatureError: Until Phase 3 implements the node.
    """
    raise planned(f"{__name__}.grade_documents", 3)


@traced
def build_context(state: RagState) -> dict[str, Any]:
    """Format the relevant chunks into a cited context and the matching sources.

    Reads ``documents`` and ``scores``. Chunks whose text repeats a higher-ranked chunk are
    dropped. Writes ``context``, the remaining chunks in rank order, each with its citation
    marker (``[1]``, ``[2]``, ...). Also writes ``sources``, one ``Source`` per marker with
    its ``score``, in the same order: ``sources[i]`` is marker ``[i + 1]``.

    The markers are local to this run. When several ``retrieve`` sub-tasks run, the main
    workflow's ``synthesize_answer`` renumbers their sources into one global numbering.

    Runs on every path. Without relevant chunks it writes an empty ``context`` and no
    ``sources``, so ``RagOutput`` is complete.

    Args:
        state: The subgraph state.

    Returns:
        The partial update ``{"context": ..., "sources": [...]}``.

    Raises:
        PlannedFeatureError: Until Phase 3 implements the node.
    """
    raise planned(f"{__name__}.build_context", 3)
