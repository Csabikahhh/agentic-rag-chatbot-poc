"""Tests for the RAG subgraph: its contract, each node on its own, and the compiled graph.

The nodes are called directly with stand-ins: a scripted chat model (``ScriptedChatModel``)
and a stand-in vector store. The compiled graph runs offline in fake mode against a small
Chroma index built under ``tmp_path`` with the hashing embeddings. That importing the modules
stays light is checked in test_imports.py.
"""

import inspect
import itertools
import logging
import threading
import time
import typing
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, get_args

import pytest
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langgraph.graph.state import CompiledStateGraph

from agentic_rag import cli
from agentic_rag.config import EmbeddingProvider, Settings
from agentic_rag.ingestion.index import IndexNotFoundError, build_index
from agentic_rag.llm import FakeRule, ScriptedChatModel
from agentic_rag.rag import graph, nodes
from agentic_rag.rag.nodes import GRADE_INSTRUCTIONS, MAX_QUERY_LENGTH, REWRITE_INSTRUCTIONS
from agentic_rag.rag.state import RagInput, RagOutput, RagState, Source

EXPECTED_RAG_NODE_NAMES = ("rewrite_query", "retrieve", "grade_documents", "build_context")

# What build_rag_graph binds to each node, following the node roles of plan section 5.2: an
# optional chat model (None skips the rewrite and the LLM grade), a lazy provider of the
# vector index, the retrieval depth and the score threshold.
RAG_NODE_DEPENDENCIES: dict[str, set[str]] = {
    "rewrite_query": {"chat_model"},
    "retrieve": {"vector_store", "top_k", "keyword_search"},
    "grade_documents": {"min_score", "chat_model", "top_k"},
    "build_context": set(),
}
RAG_DEPENDENCY_TYPES: dict[str, Any] = {
    "chat_model": BaseChatModel | None,
    "vector_store": Callable[[], VectorStore],
    "top_k": int,
    "min_score": float,
    "keyword_search": Callable[[str, int], list[Document]] | None,
}


def chunk(text: str, chunk_id: str, **metadata: Any) -> Document:
    """A retrieved chunk with the metadata the ingestion gives it."""
    return Document(
        page_content=text,
        metadata={"source": "guide.md", "title": "Guide", "chunk_id": chunk_id, **metadata},
    )


class StandInStore:
    """Answers similarity searches from a fixed result list and records the calls."""

    def __init__(self, results: list[tuple[Document, float]]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def get(self, *, limit: int, offset: int, include: list[str]) -> dict[str, Any]:
        """Expose the stored chunks to the lazy keyword index."""
        documents = [doc for doc, _ in self.results][offset : offset + limit]
        return {
            "ids": [doc.metadata["chunk_id"] for doc in documents],
            "documents": [doc.page_content for doc in documents],
            "metadatas": [doc.metadata for doc in documents],
        }

    def similarity_search_with_relevance_scores(
        self, query: str, k: int
    ) -> list[tuple[Document, float]]:
        """Return the first ``k`` results, in the order they were given."""
        self.calls.append((query, k))
        return self.results[:k]


def scripted(*rules: tuple[str, str]) -> ScriptedChatModel:
    """A scripted chat model with ``(pattern, reply)`` rules."""
    return ScriptedChatModel(rules=[FakeRule(pattern=p, reply=r) for p, r in rules])


def own_update(update: dict[str, Any]) -> dict[str, Any]:
    """A node's update without the trace event that ``traced`` adds."""
    return {key: value for key, value in update.items() if key != "trace"}


# --- the contract ---------------------------------------------------------------------------


def test_rag_node_names_are_the_four_planned_nodes_in_order() -> None:
    assert graph.RAG_NODE_NAMES == EXPECTED_RAG_NODE_NAMES
    assert set(RAG_NODE_DEPENDENCIES) == set(graph.RAG_NODE_NAMES)


def test_nodes_module_exports_exactly_the_rag_nodes() -> None:
    assert sorted(nodes.__all__) == sorted(graph.RAG_NODE_NAMES)


@pytest.mark.parametrize("name", EXPECTED_RAG_NODE_NAMES)
def test_every_rag_node_is_a_traced_function_named_after_its_node(name: str) -> None:
    node = getattr(nodes, name)

    # The traced name must equal the node name, so skip_forwarded can recognise own events.
    assert node.__name__ == name
    assert inspect.isfunction(node.__wrapped__), "the node is not wrapped with traced()"


@pytest.mark.parametrize("name", EXPECTED_RAG_NODE_NAMES)
def test_rag_nodes_take_their_state_and_explicit_dependencies(name: str) -> None:
    """The state is the only positional parameter; dependencies are keyword-only and bound."""
    node = getattr(nodes, name)
    hints = typing.get_type_hints(node)
    state_param, *dependency_params = inspect.signature(node).parameters.values()

    assert state_param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert hints[state_param.name] is RagState
    assert hints["return"] == dict[str, Any]
    assert {param.name for param in dependency_params} == RAG_NODE_DEPENDENCIES[name]
    for param in dependency_params:
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        if param.name == "keyword_search" or (name == "grade_documents" and param.name == "top_k"):
            assert param.default is None
        else:
            assert param.default is inspect.Parameter.empty
        expected_type = (
            int | None
            if name == "grade_documents" and param.name == "top_k"
            else RAG_DEPENDENCY_TYPES[param.name]
        )
        assert hints[param.name] == expected_type


def test_build_rag_graph_requires_settings_and_returns_the_typed_subgraph() -> None:
    hints = typing.get_type_hints(graph.build_rag_graph)
    (parameter,) = inspect.signature(graph.build_rag_graph).parameters.values()

    # No default: the builder binds the settings it is given and never reads get_settings().
    assert parameter.name == "settings"
    assert parameter.default is inspect.Parameter.empty
    assert hints["settings"] is Settings
    assert hints["return"] == CompiledStateGraph[RagState, None, RagInput, RagOutput]


def test_every_embedding_provider_has_a_threshold() -> None:
    assert set(graph.MIN_SCORES) == set(get_args(EmbeddingProvider))
    assert all(0.0 <= score < 1.0 for score in graph.MIN_SCORES.values())


# --- rewrite_query --------------------------------------------------------------------------


def test_rewrite_query_without_a_model_keeps_the_query() -> None:
    update = nodes.rewrite_query({"query": "Mi az a useState?"}, chat_model=None)

    assert own_update(update) == {"rewritten_query": "Mi az a useState?"}
    (event,) = update["trace"]
    assert (event.node, event.summary) == ("rewrite_query", "Search query: Mi az a useState?")


def test_rewrite_query_asks_for_an_english_search_query() -> None:
    model = scripted(("(?s)Hogyan", "fetch data on the server in the Next.js App Router"))

    update = nodes.rewrite_query({"query": "Hogyan kérek le adatot?"}, chat_model=model)

    assert update["rewritten_query"] == "fetch data on the server in the Next.js App Router"
    assert model.last_prompt == f"{REWRITE_INSTRUCTIONS}\n\nHogyan kérek le adatot?"


@pytest.mark.parametrize(
    ("reply", "query"),
    [
        (
            '<think>The user means useState.</think>\nQuery: "React useState hook"\nThis query...',
            "React useState hook",
        ),
        ("  `css :has selector`  \n", "css :has selector"),
        ("Search query: dynamic routes", "dynamic routes"),
    ],
)
def test_rewrite_query_keeps_only_the_query_of_the_reply(reply: str, query: str) -> None:
    update = nodes.rewrite_query({"query": "q"}, chat_model=scripted((".", reply)))

    assert update["rewritten_query"] == query


@pytest.mark.parametrize(
    "reply", ["", "<think>only thoughts</think>", "x" * (MAX_QUERY_LENGTH + 1)]
)
def test_an_unusable_rewrite_falls_back_to_the_query(
    reply: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="agentic_rag.rag.nodes"):
        update = nodes.rewrite_query({"query": "useState"}, chat_model=scripted((".", reply)))

    assert update["rewritten_query"] == "useState"
    assert "no usable query" in caplog.text


# --- retrieve -------------------------------------------------------------------------------


def test_retrieve_searches_with_the_rewritten_query_and_ranks_the_chunks() -> None:
    first, second = chunk("A", "a"), chunk("B", "b")
    store = StandInStore([(second, 0.4), (first, 0.9)])

    update = nodes.retrieve(
        {"query": "original", "rewritten_query": "rewritten"}, vector_store=lambda: store, top_k=4
    )

    assert store.calls == [("rewritten", 4)]
    assert own_update(update) == {"documents": [first, second], "scores": [0.9, 0.4]}
    (event,) = update["trace"]
    assert event.summary == "2 chunks retrieved, scores 0.900-0.400"
    assert event.metadata == {"chunks": 2, "top_score": 0.9}


def test_retrieve_falls_back_to_the_query() -> None:
    store = StandInStore([])

    update = nodes.retrieve({"query": "original"}, vector_store=lambda: store, top_k=2)

    assert store.calls == [("original", 2)]
    assert own_update(update) == {"documents": [], "scores": []}
    assert update["trace"][0].summary == "No chunks retrieved"


# --- grade_documents ------------------------------------------------------------------------


def graded_state(**changes: Any) -> RagState:
    """A state with three retrieved chunks, scores 0.9, 0.85 and 0.5."""
    state: dict[str, Any] = {
        "query": "Hogyan működik a useState?",
        "rewritten_query": "how does React useState work",
        "documents": [chunk("React useState", "a"), chunk("Nuxt useState", "b"), chunk("x", "c")],
        "scores": [0.9, 0.85, 0.5],
        **changes,
    }
    return typing.cast(RagState, state)


def test_grade_documents_drops_the_chunks_below_the_threshold() -> None:
    state = graded_state()

    update = nodes.grade_documents(state, min_score=0.8, chat_model=None)

    assert own_update(update) == {"documents": state["documents"][:2], "scores": [0.9, 0.85]}
    assert update["trace"][0].summary == "2 chunks kept, scores 0.900-0.850"


def test_the_llm_grade_keeps_the_chunks_the_model_lists() -> None:
    state = graded_state()
    model = scripted(("(?s)Excerpts", '{"relevant": [1, 7]}'))

    update = nodes.grade_documents(state, min_score=0.8, chat_model=model)

    assert own_update(update) == {"documents": [state["documents"][0]], "scores": [0.9]}
    prompt = model.last_prompt or ""
    assert prompt.startswith(GRADE_INSTRUCTIONS)
    assert "Question: Hogyan működik a useState?\nSearch query: how does React useState work" in (
        prompt
    )
    assert "[1] React useState\n\n[2] Nuxt useState" in prompt
    assert "[3]" not in prompt, "chunks below the threshold are not graded"


def test_an_unreadable_grade_keeps_the_chunks_above_the_threshold(
    caplog: pytest.LogCaptureFixture,
) -> None:
    state = graded_state()

    with caplog.at_level(logging.WARNING, logger="agentic_rag.rag.nodes"):
        update = nodes.grade_documents(
            state, min_score=0.8, chat_model=scripted((".", "Both are relevant."))
        )

    assert update["scores"] == [0.9, 0.85]
    assert "could not be read" in caplog.text


def test_no_chunk_above_the_threshold_means_no_llm_call() -> None:
    model = scripted((".", '{"relevant": [1]}'))

    update = nodes.grade_documents(graded_state(), min_score=0.95, chat_model=model)

    assert own_update(update) == {"documents": [], "scores": []}
    assert model.prompts == ()


# --- build_context --------------------------------------------------------------------------


def test_build_context_numbers_the_chunks_and_describes_their_sources() -> None:
    url = "https://react.dev/reference/react/useState"
    first = chunk("useState – React > Usage\n\nCall it.", "a", section="Usage", url=url)
    repeated = chunk("useState – React > Usage\n\nCall it.", "b")
    second = chunk("useState – React\n\nA Hook.", "c", page=2)

    update = nodes.build_context(
        {"query": "q", "documents": [first, repeated, second], "scores": [0.9, 0.8, 0.7]}
    )

    assert update["context"] == (
        "[1] useState – React > Usage\n\nCall it.\n\n[2] useState – React\n\nA Hook."
    )
    assert update["sources"] == [
        Source(
            chunk_id="a",
            source="guide.md",
            content=first.page_content,
            title="Guide",
            section="Usage",
            url=url,
            score=0.9,
        ),
        Source(
            chunk_id="c",
            source="guide.md",
            content=second.page_content,
            title="Guide",
            page=2,
            score=0.7,
        ),
    ]
    assert update["trace"][0].summary == "2 sources in the context"


def test_build_context_without_chunks_writes_an_empty_context() -> None:
    update = nodes.build_context({"query": "q"})

    assert own_update(update) == {"context": "", "sources": []}
    assert update["trace"][0].summary == "No relevant context"


# --- the compiled graph ---------------------------------------------------------------------


def corpus(settings: Settings) -> None:
    """Write two short pages and build their index with the fake embeddings."""
    (settings.data_dir / "has.md").write_text(
        "---\ntitle: :has()\n---\n\nThe :has() pseudo-class selects a parent element that "
        "contains a matching child element.\n",
        encoding="utf-8",
    )
    (settings.data_dir / "state.md").write_text(
        "---\ntitle: useState\n---\n\nuseState adds a state variable to a React component.\n",
        encoding="utf-8",
    )
    build_index(settings)


def with_changes(settings: Settings, **changes: Any) -> Settings:
    """Validated settings with ``changes`` applied."""
    return Settings.model_validate({**settings.model_dump(), **changes})


def test_the_graph_returns_a_cited_context_and_traces_every_node(settings: Settings) -> None:
    corpus(settings)
    rag_graph = graph.build_rag_graph(settings)

    output = rag_graph.invoke({"query": "Which pseudo-class selects a parent element?"})

    assert set(output) == {"context", "sources", "trace"}
    assert [event.node for event in output["trace"]] == list(EXPECTED_RAG_NODE_NAMES)
    assert output["context"].startswith("[1] :has()\n\nThe :has() pseudo-class selects")
    assert output["sources"][0].source == "has.md"
    assert output["sources"][0].score is not None


def test_building_does_not_open_the_index(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_index(settings: Settings) -> VectorStore:
        raise AssertionError("the index was opened while building")

    monkeypatch.setattr(graph, "load_index", no_index)

    rag_graph = graph.build_rag_graph(settings)

    assert set(rag_graph.get_graph().nodes) >= set(EXPECTED_RAG_NODE_NAMES)


def test_a_missing_index_surfaces_on_the_first_query(settings: Settings) -> None:
    rag_graph = graph.build_rag_graph(settings)

    with pytest.raises(IndexNotFoundError):
        rag_graph.invoke({"query": "anything"})


def test_concurrent_first_queries_open_the_index_once(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[int] = []
    lock = threading.Lock()

    def slow_index(settings: Settings) -> VectorStore:
        with lock:
            opened.append(1)
        time.sleep(0.05)  # long enough for the other threads to arrive
        return typing.cast(VectorStore, StandInStore([(chunk("Text", "a"), 0.9)]))

    monkeypatch.setattr(graph, "load_index", slow_index)
    rag_graph = graph.build_rag_graph(settings)

    with ThreadPoolExecutor(max_workers=8) as pool:
        outputs = list(pool.map(lambda n: rag_graph.invoke({"query": f"q{n}"}), range(8)))

    assert opened == [1]
    assert all(output["sources"][0].chunk_id == "a" for output in outputs)


def test_fake_mode_calls_no_model(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_model(settings: Settings) -> BaseChatModel:
        raise AssertionError("a chat model was created in fake mode")

    monkeypatch.setattr(graph, "get_chat_model", no_model)
    corpus(settings)

    output = graph.build_rag_graph(settings).invoke({"query": "useState"})

    assert output["trace"][0].summary == "Search query: useState"


@pytest.mark.parametrize(("grade_with_llm", "calls"), [(True, 2), (False, 1)])
def test_with_ollama_the_model_rewrites_and_optionally_grades(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, grade_with_llm: bool, calls: int
) -> None:
    corpus(settings)
    model = scripted(
        ("(?s)Excerpts", '{"relevant": [1]}'),
        ("(?s)search queries", "parent element pseudo-class"),
    )
    monkeypatch.setattr(graph, "get_chat_model", lambda settings: model)
    ollama = with_changes(settings, llm_provider="ollama", grade_with_llm=grade_with_llm)

    output = graph.build_rag_graph(ollama).invoke({"query": "Melyik szelektor?"})

    assert len(model.prompts) == calls
    assert output["trace"][0].summary == "Search query: parent element pseudo-class"
    assert output["sources"][0].source == "has.md"


def test_export_graph_draws_the_rag_subgraph(
    settings: Settings, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    assert cli.main(["export-graph", "--graph", "rag", "--format", "mermaid"]) == 0

    out = capsys.readouterr().out
    for first, second in itertools.pairwise(EXPECTED_RAG_NODE_NAMES):
        assert f"{first} --> {second};" in out
