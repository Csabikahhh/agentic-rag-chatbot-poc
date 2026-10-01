"""Shared pytest fixtures: offline settings and isolation from the developer's configuration.

Every test runs isolated: the autouse fixture removes the environment variables that
`Settings` reads, turns off loading of a `.env` file and clears the `get_settings()` cache
before and after the test, so neither the shell nor a local `.env` can change a result. A test
that reads `.env` on purpose passes `_env_file=...` explicitly.

Tests that need configuration request the `settings` fixture: fake LLM and embedding
providers (no Ollama, no model downloads) and data paths under `tmp_path`.
"""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from agentic_rag.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep the developer's environment variables and `.env` file out of every test."""
    field_names = set(Settings.model_fields)
    for name in list(os.environ):
        # Settings matches variable names case-insensitively, so drop every spelling.
        if name.lower() in field_names:
            monkeypatch.delenv(name)
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Offline settings for tests.

    The LLM and embedding providers are `fake`; `data_dir` is `tmp_path/data/raw` (created,
    empty) and `chroma_dir` is `tmp_path/data/chroma_db` (not created). Every other field has
    its default. The values are also exported as environment variables, so code that calls
    `get_settings()` (the CLI, the UI) and subprocesses started by the test see the same
    configuration.

    Returns:
        The settings, built with `_env_file=None` on top of the isolated environment.
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
