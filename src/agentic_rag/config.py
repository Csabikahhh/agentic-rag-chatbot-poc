"""Application settings, read from environment variables and an optional ``.env`` file.

All configuration goes through ``Settings`` (plan section 6); no other module reads the
environment. Each field is set by the environment variable with the upper-case field name
(``LLM_PROVIDER``, ``TOP_K``, ...). A ``.env`` file in the current working directory is read
too; it must be UTF-8, and real environment variables take precedence over it. Unknown
variables are ignored, and an empty value (``TOP_K=``) counts as unset, so the default applies.
``.env.example`` documents every variable.

Relative paths (``DATA_DIR``, ``CHROMA_DIR``) resolve against the current working directory:
the repository root when running locally, ``/app`` in the container.

Only the entry points (the CLI, the Streamlit app) and the test fixtures call
``get_settings()``; library functions take the settings as a required argument.
``describe_invalid_settings()`` names the variables behind a failed validation, and
``configure_logging()`` sets up logging for the entry points.
"""

import ipaddress
import logging
import re
from functools import lru_cache
from pathlib import Path, PurePath
from typing import Any, Literal

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from agentic_rag.errors import ConfigurationError

__all__ = [
    "EmbeddingProvider",
    "LlmProvider",
    "LogLevel",
    "Settings",
    "configure_logging",
    "describe_invalid_settings",
    "get_settings",
]

LlmProvider = Literal["ollama", "fake"]
"""Chat model backends: a local Ollama server, or the scripted offline fake."""

EmbeddingProvider = Literal["huggingface", "fake"]
"""Embedding backends: sentence-transformers via Hugging Face, or the offline fake."""

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]
"""Accepted values of ``LOG_LEVEL``."""

_ENV_FILE = ".env"

# The project's rule for CHROMA_COLLECTION, checked here so that a bad name fails at start-up
# with a clear message instead of deep inside the ingestion pipeline. It is deliberately
# stricter than Chroma: the Rust client of chromadb 1.5.9 accepts 3-512 characters from the
# same set; 63 keeps names portable across Chroma versions and deployments. Like that client,
# the rule rejects '..' and names that parse as an IPv4 address. Every name accepted here is
# valid in chromadb 1.5.9; re-check after upgrading chromadb.
_COLLECTION_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*[A-Za-z0-9]")
_COLLECTION_NAME_LENGTH = (3, 63)

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
# HTTP client libraries log every request at INFO; that drowns the application's own logs.
_NOISY_LOGGERS = ("httpx", "httpcore")


class Settings(BaseSettings):
    """Runtime configuration of the chatbot, the ingestion pipeline and the test harnesses.

    Instances are immutable (and therefore hashable, so they can key caches). Derive a changed
    copy with ``Settings.model_validate({**settings.model_dump(), **changes})``: the changes
    are validated, and because every field is passed explicitly, neither the environment nor
    ``.env`` can alter the copy. Do not use ``settings.model_copy(update=...)``, which skips
    validation.

    Relative paths resolve against the current working directory: the repository root when
    running locally, ``/app`` in the container.
    """

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        frozen=True,
        str_strip_whitespace=True,
    )

    llm_provider: LlmProvider = Field(
        default="ollama",
        description="Chat model backend: ``ollama`` (local Ollama server) or ``fake`` "
        "(scripted, deterministic and offline).",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        min_length=1,
        description="URL of the Ollama server. Inside Docker Compose use http://ollama:11434, "
        "for an Ollama on the host seen from a container http://host.docker.internal:11434.",
    )
    ollama_model: str = Field(
        default="qwen2.5:7b-instruct",
        min_length=1,
        description="Ollama chat model tag (provisional default, plan decision 4).",
    )
    ollama_num_ctx: int = Field(
        default=8192,
        ge=512,
        le=131072,
        description="Context window of the Ollama model in tokens (Ollama's ``num_ctx``), "
        "between 512 and 131072. The prompt and the answer share it, and Ollama silently "
        "truncates a prompt that does not fit. A larger window needs more GPU memory for the "
        "KV cache, for every request Ollama serves in parallel.",
    )
    ollama_timeout_s: float = Field(
        default=120.0,
        gt=0.0,
        allow_inf_nan=False,
        description="Timeout in seconds of the HTTP client for each Ollama request "
        "(connecting, sending, and every wait for response data). Keep it generous: Ollama "
        "queues concurrent requests and may load the model before the first token.",
    )
    ollama_reasoning: bool | None = Field(
        default=None,
        description="Thinking mode of reasoning models such as Qwen3.5: false turns it off, "
        "true on; empty keeps the model's default. With thinking on, every LLM call writes a "
        "long hidden reasoning first, which made Qwen3.5-4B about 15 times slower per question "
        "in the evaluation. Models without a thinking mode ignore it.",
    )
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature between 0.0 and 2.0; 0.0 keeps the answers as "
        "deterministic as the model allows.",
    )
    embedding_provider: EmbeddingProvider = Field(
        default="huggingface",
        description="Embedding backend: ``huggingface`` (sentence-transformers, runs locally) "
        "or ``fake`` (deterministic and offline).",
    )
    embedding_model: str = Field(
        default="intfloat/multilingual-e5-small",
        min_length=1,
        description="Hugging Face model id of the embedding model (provisional default, plan "
        "decision 5). The same model must be used for indexing and for querying.",
    )
    data_dir: Path = Field(
        default=Path("data/raw"),
        description="Directory with the source corpus.",
    )
    chroma_dir: Path = Field(
        default=Path("data/chroma_db"),
        description="Directory of the persistent Chroma vector index.",
    )
    chroma_collection: str = Field(
        default="documents",
        description="Chroma collection name: 3-63 characters from [A-Za-z0-9._-], starting and "
        "ending with a letter or digit, with no '..' and not an IPv4 address. The 63-character "
        "limit is the project's own, stricter than chromadb 1.5.9 (512).",
    )
    top_k: int = Field(
        default=4,
        ge=1,
        description="Number of chunks retrieved per query.",
    )
    grade_with_llm: bool = Field(
        default=True,
        description="Let the chat model drop the retrieved chunks that do not help answer the "
        "query (one extra LLM call per retrieval). Has no effect with the fake LLM provider.",
    )
    max_retries: int = Field(
        default=2,
        ge=0,
        description="Upper bound on the verify -> re-plan loop of the main workflow.",
    )
    ingest_on_start: bool = Field(
        default=True,
        description="At start-up (agentic-rag serve, the container's command), download the "
        "missing corpus sources and bring the vector index up to date: build it when it is "
        "missing, rebuild it when it was built with other embeddings.",
    )
    log_level: LogLevel = Field(
        default="INFO",
        description="Logging level of the application: DEBUG, INFO, WARNING or ERROR.",
    )

    @field_validator("llm_provider", "embedding_provider", mode="before")
    @classmethod
    def _normalize_provider(cls, value: Any) -> Any:
        """Accept provider names in any letter case (``Fake`` becomes ``fake``)."""
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, value: Any) -> Any:
        """Accept level names in any letter case (``debug`` becomes ``DEBUG``)."""
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("chroma_collection")
    @classmethod
    def _check_collection_name(cls, value: str) -> str:
        """Apply the project's collection-name rule (see the comment at the top)."""
        shortest, longest = _COLLECTION_NAME_LENGTH
        if (
            not shortest <= len(value) <= longest
            or _COLLECTION_NAME_PATTERN.fullmatch(value) is None
            or ".." in value
            or _is_ipv4_address(value)
        ):
            msg = (
                f"must be {shortest}-{longest} characters from [A-Za-z0-9._-], start and end "
                "with a letter or digit, contain no '..' and not be an IPv4 address"
            )
            raise ValueError(msg)
        return value

    def as_env(self) -> dict[str, str]:
        """Return the settings as environment-variable assignments.

        The result loads back into equal settings, so it can be written to a ``.env`` file.

        Returns:
            The upper-case variable name of every field, in declaration order, mapped to its
            value as text: booleans as ``true`` or ``false``, paths with forward slashes, and
            an unset optional value as an empty text, which loads back as unset.
        """
        return {
            name.upper(): _format_env_value(getattr(self, name)) for name in type(self).model_fields
        }


def _is_ipv4_address(value: str) -> bool:
    """Return whether a text parses as an IPv4 address, as chromadb's own check does.

    A real parse, not a dotted-quad pattern: ``10.0.0.1`` is an address, while
    ``999.999.999.999`` and ``01.2.3.4`` are not, and Chroma accepts them as names.
    """
    try:
        ipaddress.IPv4Address(value)
    except ValueError:
        return False
    return True


def _format_env_value(value: object) -> str:
    """Format one setting the way it is written in a ``.env`` file; None is an empty value."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, PurePath):
        return value.as_posix()
    return str(value)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, loaded once from the environment and ``.env``.

    Only the entry points (the CLI, the Streamlit app) and the test fixtures call it; library
    functions take the settings as an argument.

    Returns:
        The cached ``Settings`` instance. Call ``get_settings.cache_clear()`` to reload it; the
        test fixtures do this around every test. A failed load is not cached.

    Raises:
        pydantic.ValidationError: If a variable has an invalid value;
            ``describe_invalid_settings()`` names the variables.
        ConfigurationError: If ``.env`` exists but cannot be read or is not valid UTF-8.
    """
    try:
        return Settings()
    except UnicodeDecodeError as exc:
        msg = (
            f"{Path(_ENV_FILE).absolute()} is not valid UTF-8 ({exc}). Save .env as UTF-8: "
            "Windows PowerShell 5.1 writes UTF-16 with '>' and Out-File, so use "
            "'Copy-Item .env.example .env' or 'Set-Content -Encoding utf8' instead."
        )
        raise ConfigurationError(msg) from exc
    except OSError as exc:
        # Reading .env is the only file access while the settings load.
        msg = f"{Path(_ENV_FILE).absolute()} could not be read: {exc.strerror or exc}"
        raise ConfigurationError(msg) from exc


def describe_invalid_settings(error: ValidationError) -> list[tuple[str, str]]:
    """Name the invalid settings of a failed validation by their environment variables.

    The CLI prints the pairs as plain text and the Streamlit app as Markdown.

    Args:
        error: The error raised while building ``Settings``.

    Returns:
        One ``(variable, problem)`` pair per error, in the order pydantic reports them. The
        variable is the upper-case field name (``SETTINGS`` for an error that concerns no
        single field). The problem is pydantic's message, followed by the rejected value when
        it is a plain value, for example
        ``("TOP_K", "Input should be greater than or equal to 1 (got '0')")``.
    """
    problems: list[tuple[str, str]] = []
    for issue in error.errors(include_url=False):
        name = ".".join(str(part) for part in issue["loc"]).upper() or "SETTINGS"
        message = issue["msg"]
        if issue["type"] == "value_error":
            # A ValueError raised by a validator: its message reads well without the prefix.
            message = message.removeprefix("Value error, ")
        value = issue.get("input")
        if isinstance(value, str | int | float):
            message = f"{message} (got {value!r})"
        problems.append((name, message))
    return problems


def configure_logging(level: str = "INFO") -> None:
    """Configure logging for an entry point such as the CLI or the Streamlit app.

    Library modules only create loggers; this function decides where the records go. It adds a
    stderr handler unless the root logger already has one (so the handlers of pytest or of a
    host application are kept), sets the root level, and keeps the per-request INFO logs of the
    HTTP client libraries quiet unless the level is DEBUG. Calling it again only changes the
    level.

    Args:
        level: A logging level name, typically ``settings.log_level``.

    Raises:
        ValueError: If ``level`` is not a known logging level name.
    """
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        root.addHandler(handler)
    root.setLevel(level.upper())
    library_level = logging.DEBUG if root.level <= logging.DEBUG else logging.WARNING
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(library_level)
