"""Project exceptions shared by the library code and the entry points.

The CLI and the Streamlit app map these types to user-facing messages; every other exception
keeps its traceback, so a genuine bug is never mistaken for a planned gap or a user error.
"""

from typing import Final

__all__ = [
    "PLAN_REFERENCE",
    "ConfigurationError",
    "EmbeddingMismatchError",
    "IndexNotFoundError",
    "InvalidArgumentError",
    "PlannedFeatureError",
    "planned",
]

PLAN_REFERENCE: Final = "docs/project-structure-plan.md, section 8"
"""Where the phases that implement the stubs are described."""


class PlannedFeatureError(NotImplementedError):
    """A part of the prototype that a later phase of the plan implements.

    Stubs raise it through :func:`planned`. It subclasses ``NotImplementedError``, but the CLI
    and the UI catch only this type: any other ``NotImplementedError`` (for example from a
    library) is a real failure and surfaces with its traceback.
    """


class ConfigurationError(ValueError):
    """The configuration cannot be used: the settings could not be loaded (for example
    because ``.env`` is not valid UTF-8), a configuration file is invalid (``sources.toml``),
    or the settings do not match the data they open (an index built with another embedding
    model, ``EmbeddingMismatchError``).

    Invalid setting values raise ``pydantic.ValidationError`` instead; entry points handle
    both as a configuration problem (exit code 2 in the CLI).
    """


class EmbeddingMismatchError(ConfigurationError):
    """The vector index was built with another embedding provider or model.

    Raised by ``agentic_rag.ingestion.index`` (which re-exports it) when the index is opened.
    Vectors of different models are not comparable, so the index has to be rebuilt with
    ``agentic-rag ingest --rebuild``, or the settings have to match the index again. As a
    ``ConfigurationError``, the CLI reports it with exit code 2; the UI explains it in the
    assistant turn.
    """


class IndexNotFoundError(FileNotFoundError):
    """The vector index has not been built yet.

    Raised by ``agentic_rag.ingestion.index.load_index`` (which re-exports it). The message
    names the directory and the collection and tells to run ``agentic-rag ingest``. As a
    ``FileNotFoundError``, it is also caught by callers that only handle missing files.
    """


class InvalidArgumentError(ValueError):
    """A library function rejected an argument that came from the user.

    The CLI reports it like an argparse usage error (exit code 2), without a traceback.
    """


def planned(qualified_name: str, phase: int) -> PlannedFeatureError:
    """Build the error that a stub raises.

    Usage: ``raise planned(f"{__name__}.build_index", 2)``.

    Args:
        qualified_name: Dotted name of the stubbed function, method or class.
        phase: The phase of the plan that implements it.

    Returns:
        A :class:`PlannedFeatureError` with the message
        ``"<qualified_name> is planned for Phase <phase> (see docs/project-structure-plan.md,
        section 8)"``.
    """
    return PlannedFeatureError(
        f"{qualified_name} is planned for Phase {phase} (see {PLAN_REFERENCE})"
    )
