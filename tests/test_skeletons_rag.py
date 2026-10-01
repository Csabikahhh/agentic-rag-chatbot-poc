"""Tests for the skeletons of the ingestion pipeline and the RAG subgraph.

Until Phases 2 and 3 these modules are typed skeletons. The tests pin what the later phases
build on: the node names and their order, the node signatures with their explicit
dependencies, the builder's signature, the data models, the metadata keys, the planned stubs,
and that importing the modules stays light.
"""

import inspect
import json
import os
import subprocess
import sys
import typing
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langgraph.graph.state import CompiledStateGraph
from pydantic import ValidationError

from agentic_rag.config import Settings
from agentic_rag.errors import PlannedFeatureError, planned
from agentic_rag.ingestion import chunking, index, loaders
from agentic_rag.ingestion.chunking import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SEPARATORS,
    ChunkingConfig,
    ChunkMetadata,
)
from agentic_rag.ingestion.index import (
    EmbeddingMismatchError,
    IndexNotFoundError,
    IndexStats,
)
from agentic_rag.ingestion.loaders import DocumentMetadata
from agentic_rag.rag import graph, nodes
from agentic_rag.rag.state import RagInput, RagOutput, RagState, Source

EXPECTED_RAG_NODE_NAMES = ("rewrite_query", "retrieve", "grade_documents", "build_context")

# What build_rag_graph binds to each node, following the node roles of plan section 5.2: an
# optional chat model (None skips the rewrite and the LLM grade), a lazy provider of the
# vector index, the retrieval depth and the score threshold.
RAG_NODE_DEPENDENCIES: dict[str, set[str]] = {
    "rewrite_query": {"chat_model"},
    "retrieve": {"vector_store", "top_k"},
    "grade_documents": {"min_score", "chat_model"},
    "build_context": set(),
}
RAG_DEPENDENCY_TYPES: dict[str, Any] = {
    "chat_model": BaseChatModel | None,
    "vector_store": Callable[[], VectorStore],
    "top_k": int,
    "min_score": float,
}

SKELETON_MODULES = (loaders, chunking, index, nodes, graph)
LIGHT_MODULES = (
    "agentic_rag.embeddings",
    "agentic_rag.ingestion",
    "agentic_rag.ingestion.loaders",
    "agentic_rag.ingestion.chunking",
    "agentic_rag.ingestion.index",
    "agentic_rag.rag",
    "agentic_rag.rag.state",
    "agentic_rag.rag.nodes",
    "agentic_rag.rag.graph",
)
HEAVY_MODULES = (
    "torch",
    "transformers",
    "sentence_transformers",
    "langchain_huggingface",
    "langchain_text_splitters",
    "chromadb",
    "langchain_chroma",
    "streamlit",
)
DOCUMENT = Document(page_content="Some text.", metadata={"source": "guide.md"})


def unopened_index() -> VectorStore:
    """Stand in for the lazy index provider of ``retrieve``; a stub must never call it."""
    raise AssertionError("the stub opened the vector index")


# Every public function of the skeleton modules: its phase and a call with valid arguments.
STUB_CALLS: dict[str, tuple[int, Callable[[Settings], object]]] = {
    "agentic_rag.ingestion.loaders.list_corpus_files": (
        2,
        lambda settings: loaders.list_corpus_files(settings.data_dir),
    ),
    "agentic_rag.ingestion.loaders.load_file": (
        2,
        lambda settings: loaders.load_file(
            settings.data_dir / "guide.md", data_dir=settings.data_dir
        ),
    ),
    "agentic_rag.ingestion.loaders.load_documents": (2, loaders.load_documents),
    "agentic_rag.ingestion.chunking.build_splitter": (2, lambda _: chunking.build_splitter()),
    "agentic_rag.ingestion.chunking.split_documents": (
        2,
        lambda _: chunking.split_documents([DOCUMENT]),
    ),
    "agentic_rag.ingestion.index.build_index": (2, index.build_index),
    "agentic_rag.ingestion.index.load_index": (2, index.load_index),
    "agentic_rag.rag.nodes.rewrite_query": (
        3,
        lambda _: nodes.rewrite_query({"query": "q"}, chat_model=None),
    ),
    "agentic_rag.rag.nodes.retrieve": (
        3,
        lambda settings: nodes.retrieve(
            {"query": "q", "rewritten_query": "q"},
            vector_store=unopened_index,
            top_k=settings.top_k,
        ),
    ),
    "agentic_rag.rag.nodes.grade_documents": (
        3,
        lambda _: nodes.grade_documents(
            {"query": "q", "documents": [DOCUMENT], "scores": [0.5]},
            min_score=0.0,
            chat_model=None,
        ),
    ),
    "agentic_rag.rag.nodes.build_context": (
        3,
        lambda _: nodes.build_context({"query": "q", "documents": [], "scores": []}),
    ),
    "agentic_rag.rag.graph.build_rag_graph": (3, graph.build_rag_graph),
}

# Runs in a fresh interpreter: imports the modules in argv[1] and prints which of the modules
# in argv[2] got imported along the way.
_IMPORT_PROBE = """
import importlib, json, sys
for name in json.loads(sys.argv[1]):
    importlib.import_module(name)
print(json.dumps(sorted(name for name in json.loads(sys.argv[2]) if name in sys.modules)))
"""


def valid_index_stats(**changes: Any) -> dict[str, Any]:
    """Arguments of a valid ``IndexStats``, with ``changes`` applied."""
    return {
        "collection": "documents",
        "chroma_dir": Path("data/chroma_db"),
        "embedding_provider": "huggingface",
        "embedding_model": "intfloat/multilingual-e5-small",
        "files": 3,
        "documents": 12,
        "chunks": 40,
        "rebuilt": True,
        "duration_ms": 1234.5,
        **changes,
    }


# --- RAG subgraph nodes and builder ---------------------------------------------------------


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
        assert param.default is inspect.Parameter.empty, "dependencies are bound, not defaulted"
        assert hints[param.name] == RAG_DEPENDENCY_TYPES[param.name]


def test_build_rag_graph_requires_settings_and_returns_the_typed_subgraph() -> None:
    hints = typing.get_type_hints(graph.build_rag_graph)
    (parameter,) = inspect.signature(graph.build_rag_graph).parameters.values()

    # No default: the builder binds the settings it is given and never reads get_settings().
    assert parameter.name == "settings"
    assert parameter.default is inspect.Parameter.empty
    assert hints["settings"] is Settings
    assert hints["return"] == CompiledStateGraph[RagState, None, RagInput, RagOutput]


# --- stubs ----------------------------------------------------------------------------------


@pytest.mark.parametrize("qualified_name", sorted(STUB_CALLS))
def test_stub_raises_planned_feature_error_naming_its_phase(
    qualified_name: str, settings: Settings
) -> None:
    phase, call = STUB_CALLS[qualified_name]

    with pytest.raises(PlannedFeatureError) as excinfo:
        call(settings)

    assert str(excinfo.value) == str(planned(qualified_name, phase))


def test_every_public_function_of_the_skeletons_is_a_listed_stub() -> None:
    public_functions = {
        f"{module.__name__}.{name}"
        for module in SKELETON_MODULES
        for name, function in inspect.getmembers(module, inspect.isfunction)
        if function.__module__ == module.__name__ and not name.startswith("_")
    }

    assert public_functions == set(STUB_CALLS)


def test_stubs_accept_their_documented_keyword_arguments(settings: Settings) -> None:
    calls: list[Callable[[], object]] = [
        lambda: index.build_index(settings, rebuild=True),
        lambda: graph.build_rag_graph(settings=settings),
        lambda: chunking.build_splitter(ChunkingConfig()),
        lambda: chunking.split_documents([DOCUMENT], config=ChunkingConfig()),
    ]

    for call in calls:
        with pytest.raises(PlannedFeatureError):
            call()


# --- data models and metadata contracts -----------------------------------------------------


def test_index_stats_round_trips_through_json() -> None:
    stats = IndexStats(**valid_index_stats())

    assert IndexStats.model_validate_json(stats.model_dump_json()) == stats
    assert stats.chunking == ChunkingConfig()
    assert json.loads(stats.model_dump_json(indent=2))["chunks"] == 40


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("collection", ""),
        ("embedding_provider", "openai"),
        ("embedding_model", ""),
        ("files", -1),
        ("documents", -1),
        ("chunks", -1),
        ("duration_ms", -0.5),
    ],
)
def test_index_stats_rejects_invalid_values(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        IndexStats(**valid_index_stats(**{field: value}))


def test_index_stats_is_immutable() -> None:
    stats = IndexStats(**valid_index_stats())

    with pytest.raises(ValidationError):
        stats.chunks = 0  # type: ignore[misc]


def test_index_errors_extend_the_builtin_errors() -> None:
    assert issubclass(IndexNotFoundError, FileNotFoundError)
    assert issubclass(EmbeddingMismatchError, ValueError)


def test_default_chunking_is_inside_the_planned_range() -> None:
    # Plan section 5.4: start at about 800-1000 characters with 10-20 % overlap.
    assert 800 <= DEFAULT_CHUNK_SIZE <= 1000
    assert 0.10 <= DEFAULT_CHUNK_OVERLAP / DEFAULT_CHUNK_SIZE <= 0.20
    assert ChunkingConfig() == ChunkingConfig(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        separators=DEFAULT_SEPARATORS,
    )
    assert DEFAULT_SEPARATORS[-1] == ""


@pytest.mark.parametrize(
    "changes",
    [
        {"chunk_size": 100, "chunk_overlap": 100},
        {"chunk_size": 100, "chunk_overlap": 150},
        {"chunk_size": 0},
        {"chunk_overlap": -1},
        {"separators": ()},
    ],
)
def test_chunking_config_rejects_invalid_values(changes: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ChunkingConfig(**changes)


def test_chunking_config_is_immutable() -> None:
    config = ChunkingConfig()

    with pytest.raises(ValidationError):
        config.chunk_size = 10  # type: ignore[misc]


def test_document_metadata_keys_are_source_fields() -> None:
    keys = DocumentMetadata.__required_keys__ | DocumentMetadata.__optional_keys__

    assert keys == {"source", "title", "page", "section"}
    assert keys <= set(Source.model_fields)
    assert DocumentMetadata.__required_keys__ == {"source"}


def test_chunk_metadata_fills_every_source_field_except_content_and_score() -> None:
    keys = ChunkMetadata.__required_keys__ | ChunkMetadata.__optional_keys__

    assert ChunkMetadata.__required_keys__ == {"source", "chunk_id"}
    assert keys - {"start_index"} == set(Source.model_fields) - {"content", "score"}


# --- import cost ----------------------------------------------------------------------------


def test_importing_the_rag_and_ingestion_modules_loads_no_heavy_library(tmp_path: Path) -> None:
    field_names = set(Settings.model_fields)
    env = {name: value for name, value in os.environ.items() if name.lower() not in field_names}

    completed = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE, json.dumps(LIGHT_MODULES), json.dumps(HEAVY_MODULES)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=tmp_path,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == []
