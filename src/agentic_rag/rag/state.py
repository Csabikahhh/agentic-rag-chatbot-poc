"""State contracts of the RAG subgraph.

The subgraph turns one query into a cited context in four nodes (plan section 5.2):
``rewrite_query`` -> ``retrieve`` -> ``grade_documents`` -> ``build_context``. It is compiled
as ``StateGraph(RagState, input_schema=RagInput, output_schema=RagOutput)``, so:

- callers pass a :class:`RagInput` (``{"query": ...}``) and nothing else is needed to start;
- ``invoke`` returns exactly the :class:`RagOutput` keys, never the internal documents;
- :class:`RagState` is the shared memory of the four nodes and stays private to the subgraph.

Raw data lives in the state (documents, scores); prompts and citation markers are formatted
inside the nodes. Every node is wrapped with ``agentic_rag.tracing.traced``, so ``trace``
collects one event per executed node and ``RagOutput["trace"]`` gives the subgraph's
per-node latency when it runs on its own. The main graph's ``run_rag_subtask`` forwards that
list into the main ``AgentState.trace``.

:class:`Source` is the record the UI shows and the answer cites. It is shared with the main
workflow (``SubtaskResult.sources`` and ``AgentState.sources``).
"""

import operator
from typing import Annotated, Required, TypedDict

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field

from agentic_rag.tracing import TraceEvent

__all__ = ["RagInput", "RagOutput", "RagState", "Source"]


class Source(BaseModel):
    """One retrieved chunk, as shown in the UI and cited in answers.

    The record describes a chunk; its ``source`` field names the document the chunk comes
    from. Sources are immutable values: derive a changed copy with ``model_copy(update=...)``.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1, description="Stable id of the chunk in the index.")
    source: str = Field(
        min_length=1,
        description="Document the chunk comes from: its path relative to DATA_DIR with forward "
        "slashes (DocumentMetadata.source, for example guides/setup.pdf), or its URL.",
    )
    content: str = Field(description="Text of the chunk as indexed.")
    title: str | None = Field(default=None, description="Document title, when known.")
    revision: str | None = Field(default=None, description="Indexed documentation commit SHA.")
    page: int | None = Field(
        default=None, ge=1, description="1-based page number, for paged formats such as PDF."
    )
    section: str | None = Field(
        default=None, description="Heading of the section the chunk belongs to, when known."
    )
    url: str | None = Field(
        default=None, description="Public URL of the document, for a downloaded source."
    )
    score: float | None = Field(
        default=None, description="Relevance score; higher means more relevant."
    )


class RagInput(TypedDict):
    """Public input of the RAG subgraph.

    Attributes:
        query: The question or search request to answer from the knowledge base.
    """

    query: str


class RagOutput(TypedDict):
    """Public output of the RAG subgraph, returned by ``rag_graph.invoke``.

    ``build_context`` always runs, also when nothing relevant was found, so every key is
    present after a successful run.

    Attributes:
        context: Retrieved text formatted for the answer prompt, with the citation markers
            ``[1]``, ``[2]``, ... of this run; empty when nothing relevant was found.
        sources: The chunks behind ``context``, in rank order (most relevant first);
            ``sources[i]`` is marker ``[i + 1]``.
        trace: One event per executed subgraph node, in execution order.
    """

    context: str
    sources: list[Source]
    trace: list[TraceEvent]


class RagState(TypedDict, total=False):
    """Shared memory of the RAG subgraph nodes.

    Only ``query`` is required: the subgraph starts from :class:`RagInput`. ``trace`` has an
    append reducer and always exists (it starts as an empty list); the other keys are absent
    until a node writes them, so nodes read them with ``state.get(...)`` when they may be
    missing. Every other key is overwritten by each write (no reducer).

    Attributes:
        query: Original query from :class:`RagInput`.
        rewritten_query: Query used for retrieval, written by ``rewrite_query``; equal to
            ``query`` when rewriting is skipped (for example in fake mode).
        documents: Retrieved chunks with their index metadata, most relevant first, written
            by ``retrieve`` and replaced by ``grade_documents`` with the relevant subset.
        scores: Cosine similarities parallel to ``documents``; -1 means unknown for a
            keyword-only hit. Hybrid rank is independent of these scores.
        keyword_matches: Chunk ids found by lexical retrieval, eligible for LLM grading
            even when they fail the cosine threshold.
        context: Formatted context with citation markers, written by ``build_context``.
        sources: The cited chunks as :class:`Source` records in rank order, written by
            ``build_context``.
        trace: Trace events of the subgraph nodes, appended by ``traced``.
    """

    query: Required[str]
    rewritten_query: str
    documents: list[Document]
    scores: list[float]
    keyword_matches: list[str]
    context: str
    sources: list[Source]
    trace: Annotated[list[TraceEvent], operator.add]
