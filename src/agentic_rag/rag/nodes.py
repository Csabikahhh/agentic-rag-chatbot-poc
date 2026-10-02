"""Node functions of the RAG subgraph (plan section 5.2).

The four nodes run in this order and exchange data only through ``RagState``:

- ``rewrite_query`` reads ``query`` and writes ``rewritten_query``.
- ``retrieve`` reads ``rewritten_query`` (or ``query``) and writes ``documents`` and
  ``scores``.
- ``grade_documents`` reads ``documents`` and ``scores`` and writes the relevant subset of
  both.
- ``build_context`` reads ``documents`` and ``scores`` and writes ``context`` and ``sources``.

Contract:

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
  ``TraceEvent`` to ``trace``, with a one-line summary and a few counts as metadata, and
  ``trace_events_from_chunk(chunk, skip_forwarded=True)`` can tell a node's own events from
  forwarded ones.
- ``build_context`` runs on every path, also when nothing relevant was found (``context`` is
  then empty and ``sources`` is ``[]``), because ``RagOutput`` is total.
- ``Source`` fields come from ``ChunkMetadata``; ``page`` is already 1-based there.

Model calls: the corpus is English documentation, while questions may be Hungarian, so
``rewrite_query`` always asks for an English search query (README, *Cross-lingual
retrieval*). ``grade_documents`` grades all chunks of a run in one structured-output call
(:class:`RelevanceGrade`), not one call per chunk. An answer of the model that cannot be used
(an empty rewrite, a grade that is not valid JSON) falls back to the input: the original
query, or the chunks that passed the threshold, and a warning is logged. Errors
of the model server itself (connection, timeout) propagate: the main workflow retries the
whole sub-task.
"""

import logging
import re
from collections.abc import Callable, Sequence
from typing import Any, Final

from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agentic_rag.rag.state import RagState, Source
from agentic_rag.tracing import traced

__all__ = ["build_context", "grade_documents", "retrieve", "rewrite_query"]

logger = logging.getLogger(__name__)

REWRITE_INSTRUCTIONS: Final = (
    "You turn questions about web frontend development into search queries for an English "
    "documentation index: MDN Web Docs (HTML, CSS, JavaScript, accessibility), React, Vue, "
    "Next.js, Nuxt and TypeScript.\n"
    "Write one short English search query with the key terms of the question. Keep the names "
    "of APIs, hooks, components, CSS properties and selectors exactly as the documentation "
    "writes them, and keep the framework name when the question names one. Translate a "
    "question in another language into English.\n"
    "Answer with the query only, on one line, without quotes or explanations."
)
"""System prompt of ``rewrite_query``."""

GRADE_INSTRUCTIONS: Final = (
    "You decide which documentation excerpts help to answer a question.\n"
    "An excerpt is relevant when it explains, defines or shows an example of what the "
    "question asks about, even if it answers only a part of the question. An excerpt about "
    "another framework, or about a different API with a "
    "similar name, is not relevant: React's useState hook and Nuxt's useState composable are "
    "different things.\n"
    'Reply with a JSON object of the form {"relevant": [1, 3]} listing the numbers of the '
    'relevant excerpts, or {"relevant": []} when none is relevant.'
)
"""System prompt of ``grade_documents``."""

MAX_QUERY_LENGTH: Final = 300
"""Longest rewrite that is used; a longer reply is an explanation, not a query."""

# A reasoning model may put its thoughts in the reply; only the text after them is the answer.
_THINKING: Final = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_QUERY_PREFIX: Final = re.compile(r"^(?:search\s+)?query\s*:\s*", re.IGNORECASE)
_QUOTES: Final = "\"'`“”„"


class RelevanceGrade(BaseModel):
    """Structured reply of the LLM grade: the 1-based numbers of the relevant excerpts."""

    model_config = ConfigDict(extra="ignore")

    relevant: list[int] = Field(
        description="Numbers of the excerpts that help to answer the question."
    )


@traced(summarize=lambda update: f"Search query: {update.get('rewritten_query', '')}")
def rewrite_query(state: RagState, *, chat_model: BaseChatModel | None) -> dict[str, Any]:
    """Rewrite the query into a standalone English search query.

    Reads ``query``. Writes ``rewritten_query``: a short, self-contained English search query
    written by ``chat_model`` (see :data:`REWRITE_INSTRUCTIONS`), or ``query`` unchanged when
    ``chat_model`` is None. The reply is reduced to its first line, without quotes, a
    ``Query:`` prefix or the thinking block of a reasoning model; an empty reply, or one
    longer than :data:`MAX_QUERY_LENGTH`, keeps ``query`` and logs a warning.

    Args:
        state: The subgraph state.
        chat_model: The chat model that rewrites the query, or None to skip the rewrite.
            ``build_rag_graph`` passes None with the fake LLM provider, because plan section
            5.2 skips the rewrite in fake mode.

    Returns:
        The partial update ``{"rewritten_query": ...}``.
    """
    query = state["query"]
    if chat_model is None:
        return {"rewritten_query": query}
    reply = chat_model.invoke(_rewrite_messages(query))
    rewritten = _clean_query(str(reply.text))
    if rewritten is None:
        logger.warning("The query rewrite gave no usable query; searching for %r", query)
        return {"rewritten_query": query}
    return {"rewritten_query": rewritten}


@traced(
    summarize=lambda update: _summarize_ranked("retrieved", update),
    metadata=lambda update: {
        "chunks": len(update.get("documents", [])),
        "top_score": max(update.get("scores", []), default=None),
    },
)
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
    """
    query = state.get("rewritten_query") or state["query"]
    results = vector_store().similarity_search_with_relevance_scores(query, k=top_k)
    ranked = sorted(results, key=lambda pair: pair[1], reverse=True)
    return {
        "documents": [document for document, _ in ranked],
        "scores": [float(score) for _, score in ranked],
    }


@traced(
    summarize=lambda update: _summarize_ranked("kept", update),
    metadata=lambda update: {"chunks": len(update.get("documents", []))},
)
def grade_documents(
    state: RagState, *, min_score: float, chat_model: BaseChatModel | None
) -> dict[str, Any]:
    """Drop the retrieved chunks that are not relevant to the query.

    Reads ``documents`` and ``scores``, and ``query`` and ``rewritten_query`` for the LLM
    grade. Writes ``documents`` and ``scores`` again with the relevant subset in rank order.
    Chunks that score below ``min_score`` are dropped first. Then, if ``chat_model`` is set
    and chunks remain, the model grades them in one call (:data:`GRADE_INSTRUCTIONS`,
    :class:`RelevanceGrade`) and the chunks it does not list are dropped too (plan section
    5.2). A grade that is not valid JSON keeps every chunk that passed the threshold and logs
    a warning. Both lists may become empty.

    Args:
        state: The subgraph state.
        min_score: The lowest relevance score a chunk may have and still be kept.
        chat_model: The chat model that grades the chunks above the threshold, or None to
            keep all of them. ``build_rag_graph`` passes None with the fake LLM provider and
            when ``GRADE_WITH_LLM`` is off.

    Returns:
        The partial update ``{"documents": [...], "scores": [...]}``.
    """
    pairs = zip(state.get("documents", []), state.get("scores", []), strict=True)
    kept = [(document, score) for document, score in pairs if score >= min_score]
    if chat_model is not None and kept:
        relevant = _llm_grade(chat_model, state, [document for document, _ in kept])
        if relevant is not None:
            kept = [pair for number, pair in enumerate(kept, start=1) if number in relevant]
    return {
        "documents": [document for document, _ in kept],
        "scores": [score for _, score in kept],
    }


@traced(
    summarize=lambda update: (
        f"{len(update.get('sources', []))} sources in the context"
        if update.get("sources")
        else "No relevant context"
    ),
    metadata=lambda update: {"sources": len(update.get("sources", []))},
)
def build_context(state: RagState) -> dict[str, Any]:
    """Format the relevant chunks into a cited context and the matching sources.

    Reads ``documents`` and ``scores``. Chunks whose text repeats a higher-ranked chunk are
    dropped. Writes ``context``, the remaining chunks in rank order, each with its citation
    marker (``[1]``, ``[2]``, ...) in front of its text; the text starts with the chunk's
    context line (title and section, see ``agentic_rag.ingestion.chunking``), and the chunks
    are separated by a blank line. Also writes ``sources``, one ``Source`` per marker with its
    ``score``, in the same order: ``sources[i]`` is marker ``[i + 1]``.

    The markers are local to this run. When several ``retrieve`` sub-tasks run, the main
    workflow's ``synthesize_answer`` renumbers their sources into one global numbering.

    Runs on every path. Without relevant chunks it writes an empty ``context`` and no
    ``sources``, so ``RagOutput`` is complete.

    Args:
        state: The subgraph state.

    Returns:
        The partial update ``{"context": ..., "sources": [...]}``.
    """
    seen: set[str] = set()
    sources: list[Source] = []
    for document, score in zip(state.get("documents", []), state.get("scores", []), strict=True):
        text = document.page_content.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        sources.append(_source(document, score))
    context = "\n\n".join(
        f"[{number}] {source.content.strip()}" for number, source in enumerate(sources, start=1)
    )
    return {"context": context, "sources": sources}


def _rewrite_messages(query: str) -> list[BaseMessage]:
    """The prompt of ``rewrite_query``."""
    return [SystemMessage(REWRITE_INSTRUCTIONS), HumanMessage(query)]


def _clean_query(reply: str) -> str | None:
    """Reduce a rewrite reply to its query; None when nothing usable is left."""
    text = _THINKING.sub("", reply)
    line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    line = _QUERY_PREFIX.sub("", line).strip().strip(_QUOTES).strip()
    if not line or len(line) > MAX_QUERY_LENGTH:
        return None
    return line


def _llm_grade(
    chat_model: BaseChatModel, state: RagState, documents: Sequence[Document]
) -> set[int] | None:
    """Ask the model which excerpts are relevant; None when its reply cannot be used."""
    query = state["query"]
    rewritten = state.get("rewritten_query") or query
    question = query if rewritten == query else f"{query}\nSearch query: {rewritten}"
    excerpts = "\n\n".join(
        f"[{number}] {document.page_content.strip()}"
        for number, document in enumerate(documents, start=1)
    )
    messages = [
        SystemMessage(GRADE_INSTRUCTIONS),
        HumanMessage(f"Question: {question}\n\nExcerpts:\n\n{excerpts}"),
    ]
    try:
        grade = chat_model.with_structured_output(RelevanceGrade).invoke(messages)
        relevant = RelevanceGrade.model_validate(grade).relevant
    except (OutputParserException, ValidationError) as exc:
        logger.warning("The relevance grade could not be read; keeping every chunk: %s", exc)
        return None
    return {number for number in relevant if 1 <= number <= len(documents)}


def _source(document: Document, score: float) -> Source:
    """The ``Source`` record of a retrieved chunk."""
    metadata = document.metadata
    return Source(
        chunk_id=str(metadata.get("chunk_id") or document.id or ""),
        source=str(metadata.get("source", "")),
        content=document.page_content,
        title=metadata.get("title"),
        page=metadata.get("page"),
        section=metadata.get("section"),
        url=metadata.get("url"),
        score=score,
    )


def _summarize_ranked(verb: str, update: dict[str, Any]) -> str:
    """Trace summary of a node that writes ranked chunks."""
    scores = update.get("scores", [])
    if not scores:
        return f"No chunks {verb}"
    noun = "chunk" if len(scores) == 1 else "chunks"
    return f"{len(scores)} {noun} {verb}, scores {max(scores):.3f}-{min(scores):.3f}"
