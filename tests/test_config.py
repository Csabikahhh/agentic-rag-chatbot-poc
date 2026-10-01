"""Tests for agentic_rag.config: defaults, environment and .env handling, validation, logging."""

import logging
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_rag import config
from agentic_rag.config import (
    Settings,
    configure_logging,
    describe_invalid_settings,
    get_settings,
)
from agentic_rag.errors import ConfigurationError

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_EXAMPLE = REPO_ROOT / ".env.example"

# Read at import time, before the autouse fixture of conftest.py turns off .env loading.
CONFIGURED_ENV_FILE = Settings.model_config.get("env_file")

EXPECTED_DEFAULTS = {
    "llm_provider": "ollama",
    "ollama_base_url": "http://localhost:11434",
    "ollama_model": "qwen2.5:7b-instruct",
    "ollama_num_ctx": 8192,
    "ollama_timeout_s": 120.0,
    "llm_temperature": 0.0,
    "embedding_provider": "huggingface",
    "embedding_model": "intfloat/multilingual-e5-small",
    "data_dir": Path("data/raw"),
    "chroma_dir": Path("data/chroma_db"),
    "chroma_collection": "documents",
    "top_k": 4,
    "max_retries": 2,
    "ingest_on_start": True,
    "log_level": "INFO",
}


def read_env_file(path: Path) -> dict[str, str]:
    """Parse the active KEY=value lines of an env file, skipping comments and blank lines."""
    assignments: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition("=")
        assert separator, f"not a KEY=value line: {line!r}"
        assignments[key.strip()] = value.strip()
    return assignments


def read_dotenv_from(directory: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Let ``get_settings()`` read ``directory/.env``, as it does outside the tests."""
    monkeypatch.chdir(directory)
    monkeypatch.setitem(Settings.model_config, "env_file", CONFIGURED_ENV_FILE)


def test_fields_and_defaults_are_the_expected_ones() -> None:
    assert list(Settings.model_fields) == list(EXPECTED_DEFAULTS)
    assert Settings(_env_file=None).model_dump() == EXPECTED_DEFAULTS


def test_environment_variables_override_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    overrides = {
        "LLM_PROVIDER": "fake",
        "OLLAMA_BASE_URL": "http://ollama:11434",
        "OLLAMA_MODEL": "llama3.1:8b",
        "OLLAMA_NUM_CTX": "16384",
        "OLLAMA_TIMEOUT_S": "30.5",
        "LLM_TEMPERATURE": "0.7",
        "EMBEDDING_PROVIDER": "fake",
        "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5",
        "DATA_DIR": str(tmp_path / "corpus"),
        "CHROMA_DIR": str(tmp_path / "index"),
        "CHROMA_COLLECTION": "docs_v1.0-test",
        "TOP_K": "7",
        "MAX_RETRIES": "0",
        "INGEST_ON_START": "false",
        "LOG_LEVEL": "WARNING",
    }
    for name, value in overrides.items():
        monkeypatch.setenv(name, value)

    settings = Settings(_env_file=None)

    assert settings.model_dump() == {
        "llm_provider": "fake",
        "ollama_base_url": "http://ollama:11434",
        "ollama_model": "llama3.1:8b",
        "ollama_num_ctx": 16384,
        "ollama_timeout_s": 30.5,
        "llm_temperature": 0.7,
        "embedding_provider": "fake",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "data_dir": tmp_path / "corpus",
        "chroma_dir": tmp_path / "index",
        "chroma_collection": "docs_v1.0-test",
        "top_k": 7,
        "max_retries": 0,
        "ingest_on_start": False,
        "log_level": "WARNING",
    }


def test_choice_values_are_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", " Fake ")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "HuggingFace")
    monkeypatch.setenv("LOG_LEVEL", "debug")

    settings = Settings(_env_file=None)

    assert (settings.llm_provider, settings.embedding_provider, settings.log_level) == (
        "fake",
        "huggingface",
        "DEBUG",
    )


def test_empty_values_fall_back_to_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOP_K", "")
    monkeypatch.setenv("OLLAMA_MODEL", "")

    settings = Settings(_env_file=None)

    assert settings.top_k == 4
    assert settings.ollama_model == "qwen2.5:7b-instruct"


def test_keyword_arguments_override_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOP_K", "7")

    assert Settings(_env_file=None, top_k=2).top_k == 2


# --- validation ---------------------------------------------------------------------------

# Boundary cases that the CHROMA_COLLECTION rule accepts.
VALID_COLLECTION_NAMES = [
    "abc",
    "x" * 63,
    "docs_v1.0-test",
    # Not IPv4 addresses, so chromadb accepts them as names; a dotted-quad pattern would not.
    "999.999.999.999",
    "256.1.1.1",
    "01.2.3.4",
    "1.2.3",
    "1.2.3.4.5",
]
# Names that parse as IPv4 addresses: chromadb rejects them, so the rule rejects them too.
IPV4_COLLECTION_NAMES = ["10.0.0.1", "255.255.255.255"]


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("LLM_PROVIDER", "openai"),
        ("EMBEDDING_PROVIDER", "openai"),
        ("OLLAMA_MODEL", "   "),
        ("OLLAMA_NUM_CTX", "511"),
        ("OLLAMA_NUM_CTX", "131073"),
        ("OLLAMA_NUM_CTX", "8k"),
        ("OLLAMA_TIMEOUT_S", "0"),
        ("OLLAMA_TIMEOUT_S", "-5"),
        ("OLLAMA_TIMEOUT_S", "inf"),
        ("OLLAMA_TIMEOUT_S", "nan"),
        ("LLM_TEMPERATURE", "-0.1"),
        ("LLM_TEMPERATURE", "2.5"),
        ("TOP_K", "0"),
        ("TOP_K", "many"),
        ("MAX_RETRIES", "-1"),
        ("INGEST_ON_START", "maybe"),
        ("LOG_LEVEL", "TRACE"),
        ("CHROMA_COLLECTION", "ab"),
        ("CHROMA_COLLECTION", "-documents"),
        ("CHROMA_COLLECTION", "my documents"),
        ("CHROMA_COLLECTION", "docs..v1"),
        ("CHROMA_COLLECTION", "x" * 64),
        *(("CHROMA_COLLECTION", name) for name in IPV4_COLLECTION_NAMES),
    ],
)
def test_invalid_values_are_rejected(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)

    assert [error["loc"] for error in excinfo.value.errors()] == [(name.lower(),)]


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("OLLAMA_NUM_CTX", "512", 512),
        ("OLLAMA_NUM_CTX", "131072", 131072),
        ("OLLAMA_TIMEOUT_S", "0.5", 0.5),
        ("OLLAMA_TIMEOUT_S", "600", 600.0),
    ],
)
def test_ollama_limits_accept_their_bounds(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str, expected: float
) -> None:
    monkeypatch.setenv(name, value)

    assert getattr(Settings(_env_file=None), name.lower()) == expected


@pytest.mark.parametrize("name", VALID_COLLECTION_NAMES)
def test_valid_collection_names_are_accepted(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.setenv("CHROMA_COLLECTION", name)

    assert Settings(_env_file=None).chroma_collection == name


def test_the_collection_rule_agrees_with_the_installed_chromadb(tmp_path: Path) -> None:
    # The rule promises that chromadb accepts every name it accepts, and it copies chromadb's
    # IPv4 rule. Checking the boundary cases against the installed client turns "re-check
    # after upgrading chromadb" into a failing test instead of a late ingestion error.
    chromadb = pytest.importorskip("chromadb")
    from chromadb.config import Settings as ChromaSettings
    from chromadb.errors import InvalidArgumentError as ChromaInvalidArgumentError

    client = chromadb.PersistentClient(
        path=str(tmp_path / "chroma"), settings=ChromaSettings(anonymized_telemetry=False)
    )

    for name in VALID_COLLECTION_NAMES:
        assert client.create_collection(name, embedding_function=None).name == name
    for name in IPV4_COLLECTION_NAMES:
        with pytest.raises(ChromaInvalidArgumentError):
            client.create_collection(name, embedding_function=None)


def test_collection_name_error_states_the_project_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHROMA_COLLECTION", "10.0.0.1")

    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)

    [(name, problem)] = describe_invalid_settings(excinfo.value)
    assert name == "CHROMA_COLLECTION"
    assert problem.startswith("must be 3-63 characters from [A-Za-z0-9._-]")
    assert problem.endswith("not be an IPv4 address (got '10.0.0.1')")


def test_describe_invalid_settings_names_the_variables_and_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TOP_K", "0")
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)

    # The problems are pydantic's own messages, so compare with them instead of their wording.
    provider_error, top_k_error = excinfo.value.errors()
    assert describe_invalid_settings(excinfo.value) == [
        ("LLM_PROVIDER", f"{provider_error['msg']} (got 'openai')"),
        ("TOP_K", f"{top_k_error['msg']} (got '0')"),
    ]


def test_describe_invalid_settings_has_a_name_for_errors_without_a_field() -> None:
    error = ValidationError.from_exception_data(
        "Settings", [{"type": "missing", "loc": (), "input": {}}]
    )

    [(name, problem)] = describe_invalid_settings(error)

    assert name == "SETTINGS"
    assert problem == error.errors()[0]["msg"]


# --- immutability and changed copies ------------------------------------------------------


def test_settings_are_immutable_and_hashable() -> None:
    settings = Settings(_env_file=None)

    with pytest.raises(ValidationError):
        settings.top_k = 8  # type: ignore[misc]

    assert hash(settings) == hash(Settings(_env_file=None))


def test_changed_copies_are_validated_and_ignore_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(_env_file=None, top_k=3)
    monkeypatch.setenv("OLLAMA_MODEL", "from-the-environment")
    monkeypatch.setenv("TOP_K", "not-a-number")

    variant = Settings.model_validate({**settings.model_dump(), "max_retries": 5})

    assert variant.model_dump() == {**settings.model_dump(), "max_retries": 5}
    with pytest.raises(ValidationError):
        Settings.model_validate({**settings.model_dump(), "top_k": 0})
    with pytest.raises(ValidationError):
        Settings.model_validate({**settings.model_dump(), "chroma_collection": "10.0.0.1"})
    # Why the docstring recommends model_validate: model_copy skips every validator.
    assert settings.model_copy(update={"top_k": 0}).top_k == 0


# --- .env and get_settings() --------------------------------------------------------------


def test_dotenv_file_in_the_working_directory_is_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / ".env").write_text(
        "# local overrides\nLLM_PROVIDER=fake\nTOP_K=9\nSOME_OTHER_TOOL_SETTING=1\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert CONFIGURED_ENV_FILE == ".env"
    settings = Settings(_env_file=CONFIGURED_ENV_FILE)

    # Unknown variables (SOME_OTHER_TOOL_SETTING) are ignored rather than rejected.
    assert (settings.llm_provider, settings.top_k) == ("fake", 9)


def test_environment_variables_take_precedence_over_dotenv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("TOP_K=9\n", encoding="utf-8")
    monkeypatch.setenv("TOP_K", "3")

    assert Settings(_env_file=env_file).top_k == 3


def test_tests_never_read_a_local_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("LLM_PROVIDER=fake\nTOP_K=9\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert Settings(_env_file=None).top_k == 4
    # The autouse fixture also turns off .env loading for code that calls Settings() itself.
    assert Settings().top_k == 4
    assert get_settings().top_k == 4


def test_settings_fixture_is_offline_and_shared_with_get_settings(
    settings: Settings, tmp_path: Path
) -> None:
    assert (settings.llm_provider, settings.embedding_provider) == ("fake", "fake")
    assert settings.data_dir == tmp_path / "data" / "raw"
    assert settings.data_dir.is_dir()
    assert settings.chroma_dir == tmp_path / "data" / "chroma_db"
    assert not settings.chroma_dir.exists()
    assert settings.top_k == EXPECTED_DEFAULTS["top_k"]
    assert get_settings() == settings


def test_get_settings_is_cached_until_cleared(monkeypatch: pytest.MonkeyPatch) -> None:
    first = get_settings()
    monkeypatch.setenv("TOP_K", "6")

    assert get_settings() is first
    assert get_settings().top_k == 4

    get_settings.cache_clear()
    assert get_settings().top_k == 6


@pytest.mark.parametrize(
    ("content", "encoding"),
    [
        # What `echo LLM_PROVIDER=fake > .env` writes in Windows PowerShell 5.1.
        pytest.param("LLM_PROVIDER=fake\r\n", "utf-16", id="utf-16-with-bom"),
        # What Set-Content writes there by default: the ANSI code page.
        pytest.param("OLLAMA_MODEL=llamá\r\n", "cp1250", id="ansi-code-page"),
    ],
)
def test_dotenv_that_is_not_utf8_is_a_configuration_error(
    content: str, encoding: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / ".env").write_bytes(content.encode(encoding))
    read_dotenv_from(tmp_path, monkeypatch)

    with pytest.raises(ConfigurationError, match=r"Save \.env as UTF-8") as excinfo:
        get_settings()

    assert ".env is not valid UTF-8" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, UnicodeDecodeError)


def test_dotenv_with_a_utf8_bom_is_read(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Set-Content -Encoding utf8 in Windows PowerShell 5.1 writes a BOM, which is fine.
    (tmp_path / ".env").write_text("TOP_K=9\n", encoding="utf-8-sig")
    read_dotenv_from(tmp_path, monkeypatch)

    assert get_settings().top_k == 9


def test_unreadable_dotenv_is_a_configuration_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def unreadable_dotenv() -> Settings:
        raise PermissionError(13, "Permission denied", ".env")

    monkeypatch.setattr(config, "Settings", unreadable_dotenv)

    with pytest.raises(ConfigurationError, match="could not be read: Permission denied"):
        get_settings()


def test_a_failed_load_is_not_cached(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_bytes("TOP_K=9\n".encode("utf-16"))
    read_dotenv_from(tmp_path, monkeypatch)
    with pytest.raises(ConfigurationError):
        get_settings()

    env_file.write_text("TOP_K=9\n", encoding="utf-8")

    assert get_settings().top_k == 9


# --- as_env and .env.example --------------------------------------------------------------


def test_as_env_round_trips_through_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = Settings(
        _env_file=None, ingest_on_start=False, data_dir=tmp_path / "corpus", llm_temperature=0.25
    )

    env = settings.as_env()

    assert list(env) == [name.upper() for name in EXPECTED_DEFAULTS]
    assert env["INGEST_ON_START"] == "false"
    assert env["DATA_DIR"] == (tmp_path / "corpus").as_posix()
    assert env["CHROMA_DIR"] == "data/chroma_db"
    assert env["LLM_TEMPERATURE"] == "0.25"
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert Settings(_env_file=None) == settings


def test_env_example_lists_every_setting_with_its_default() -> None:
    assignments = read_env_file(ENV_EXAMPLE)

    assert list(assignments.items()) == list(Settings(_env_file=None).as_env().items())
    assert Settings(_env_file=ENV_EXAMPLE) == Settings(_env_file=None)


def test_env_example_comments_every_variable_and_shows_the_run_modes() -> None:
    lines = ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()

    for index, line in enumerate(lines):
        if line and not line.startswith("#"):
            assert lines[index - 1].startswith("#"), f"no comment above {line!r}"
    commented = {line.lstrip("#").strip() for line in lines if line.startswith("#")}
    assert {
        "OLLAMA_BASE_URL=http://ollama:11434",
        "OLLAMA_BASE_URL=http://host.docker.internal:11434",
        "LLM_PROVIDER=fake",
        "EMBEDDING_PROVIDER=fake",
    } <= commented


# --- logging ------------------------------------------------------------------------------
# root_logger comes from conftest.py, whose autouse fixture restores the logging setup.


def test_configure_logging_adds_one_handler_and_sets_levels(root_logger: logging.Logger) -> None:
    root_logger.handlers[:] = []

    configure_logging("info")
    configure_logging("INFO")

    assert len(root_logger.handlers) == 1
    assert root_logger.level == logging.INFO
    assert logging.getLogger("httpx").level == logging.WARNING

    configure_logging("DEBUG")
    assert root_logger.level == logging.DEBUG
    assert logging.getLogger("httpcore").level == logging.DEBUG


def test_configure_logging_keeps_existing_handlers(root_logger: logging.Logger) -> None:
    existing = logging.NullHandler()
    root_logger.handlers[:] = [existing]

    configure_logging("ERROR")

    assert root_logger.handlers == [existing]
    assert root_logger.level == logging.ERROR


def test_configure_logging_rejects_unknown_levels(root_logger: logging.Logger) -> None:
    with pytest.raises(ValueError, match="LOUD"):
        configure_logging("loud")
