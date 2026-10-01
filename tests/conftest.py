"""Shared pytest fixtures and constants: offline settings, isolation and the heavy-module lists.

Every test runs isolated: the autouse fixture removes the environment variables that
``Settings`` reads, turns off loading of a ``.env`` file and clears the ``get_settings()``
cache before and after the test, so neither the shell nor a local ``.env`` can change a
result. A test that reads ``.env`` on purpose passes ``_env_file=...`` explicitly. A second
autouse fixture restores the logging configuration that ``configure_logging()`` changes.

Tests that need configuration request the ``settings`` fixture: fake LLM and embedding
providers (no Ollama, no model downloads) and data paths under ``tmp_path``.

``HEAVY_MODULES`` and ``GRAPH_RUNTIME_MODULES`` name the libraries that importing the project
must not load. Test modules import them with ``from conftest import ...``; a module that needs
a different set derives it with ``heavy_modules(allow=..., add=...)`` instead of writing its
own list (the UI, for example, allows ``streamlit``).
"""

import logging
import os
from collections.abc import Iterable, Iterator
from pathlib import Path

import pytest

from agentic_rag.config import Settings, get_settings

HEAVY_MODULES: frozenset[str] = frozenset(
    {
        "chromadb",
        "langchain_chroma",
        "langchain_huggingface",
        "langchain_text_splitters",
        "sentence_transformers",
        "streamlit",
        "torch",
        "transformers",
    }
)
"""Top-level packages that no project module may load at import time.

The model, vector-store, text-splitting and UI libraries: each costs seconds to import or
pulls in a model runtime. The project imports them lazily, inside the functions that use them.
"""

GRAPH_RUNTIME_MODULES: frozenset[str] = frozenset({"langchain_core", "langgraph", "langsmith"})
"""Top-level packages of the graph runtime.

The light modules (``tracing``, ``agent.types``, ``reports``, the evaluation records and
metrics, the load-test statistics) must not load them either, so that reading a committed
report or the question set stays cheap.
"""


def heavy_modules(*, allow: Iterable[str] = (), add: Iterable[str] = ()) -> frozenset[str]:
    """Return ``HEAVY_MODULES`` without ``allow`` and with ``add``: the set a test needs.

    Args:
        allow: Heavy modules this test permits, e.g. ``streamlit`` for the UI. Each must be in
            ``HEAVY_MODULES``, so a typo cannot silently allow nothing.
        add: Further top-level packages this test forbids, e.g. ``GRAPH_RUNTIME_MODULES``.

    Returns:
        The top-level package names that must not be loaded.

    Raises:
        ValueError: If ``allow`` names a module that is not in ``HEAVY_MODULES``.
    """
    allowed = frozenset(allow)
    unknown = allowed - HEAVY_MODULES
    if unknown:
        msg = f"not in HEAVY_MODULES: {sorted(unknown)}"
        raise ValueError(msg)
    return (HEAVY_MODULES - allowed) | frozenset(add)


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep the developer's environment variables and ``.env`` file out of every test."""
    field_names = set(Settings.model_fields)
    for name in list(os.environ):
        # Settings matches variable names case-insensitively, so drop every spelling.
        if name.lower() in field_names:
            monkeypatch.delenv(name)
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    """Put the logging configuration back after every test.

    ``configure_logging()`` (called by the CLI, the UI and the logging tests) sets the root
    level, may add a root handler and sets the levels of the HTTP libraries' loggers. Afterwards
    the root logger gets its level and handlers back, every logger that existed gets its level
    back, and a logger created during the test is reset to ``NOTSET``.
    """
    root = logging.getLogger()
    saved_root = root.level, list(root.handlers)
    saved_levels = {
        name: logger.level
        for name, logger in logging.Logger.manager.loggerDict.items()
        if isinstance(logger, logging.Logger)
    }
    yield
    root.setLevel(saved_root[0])
    root.handlers[:] = saved_root[1]
    for name, logger in list(logging.Logger.manager.loggerDict.items()):
        level = saved_levels.get(name, logging.NOTSET)
        # setLevel clears the level cache of every logger: call it only for a changed level.
        if isinstance(logger, logging.Logger) and logger.level != level:
            logger.setLevel(level)


@pytest.fixture
def root_logger() -> logging.Logger:
    """The root logger; ``_restore_logging`` restores its level and handlers after the test."""
    return logging.getLogger()


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Offline settings for tests.

    The LLM and embedding providers are ``fake``; ``data_dir`` is ``tmp_path/data/raw``
    (created, empty) and ``chroma_dir`` is ``tmp_path/data/chroma_db`` (not created). Every
    other field has its default. The values are also exported as environment variables, so
    code that calls ``get_settings()`` (the CLI, the UI) and subprocesses started by the test
    see the same configuration.

    Returns:
        The settings, built with ``_env_file=None`` on top of the isolated environment.
    """
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True)
    overrides = {
        "LLM_PROVIDER": "fake",
        "EMBEDDING_PROVIDER": "fake",
        "DATA_DIR": str(data_dir),
        "CHROMA_DIR": str(tmp_path / "data" / "chroma_db"),
    }
    for name, value in overrides.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    return Settings(_env_file=None)
