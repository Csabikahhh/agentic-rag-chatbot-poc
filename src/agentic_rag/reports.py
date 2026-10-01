"""Shared scaffolding of the run reports: the results directory and the :class:`RunReport` base.

``agentic-rag eval`` writes an ``EvalReport`` (``agentic_rag.evaluation.runner``) and
``agentic-rag loadtest`` writes a ``LoadTestReport`` (``agentic_rag.loadtest.runner``). Both
extend :class:`RunReport`, which records when the run finished and the settings it ran with,
so every committed result file names the configuration that produced it (plan section 10).

The module needs only Pydantic and the settings: loading a committed report loads neither
LangGraph nor a model library.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

from agentic_rag.config import Settings

__all__ = ["RESULTS_DIR", "RunReport"]

RESULTS_DIR: Final = Path("data/eval/results")
"""Report directory of ``agentic-rag eval`` and ``agentic-rag loadtest`` when
``--output-dir`` is not given. Like the paths in ``Settings``, it is relative to the working
directory: the repository root when running locally, ``/app`` in the container. The reports of
the final runs are committed."""


def _utc_now() -> datetime:
    """Return the current time in UTC, the default timestamp of a report."""
    return datetime.now(UTC)


class RunReport(BaseModel):
    """Base of the report models: when the run finished and the settings it ran with.

    Subclasses add the results of their kind of run. Reports are immutable values, and they
    reject NaN and infinite numbers, so every report serializes to valid JSON and loads back
    unchanged.
    """

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    created_at: AwareDatetime = Field(
        default_factory=_utc_now,
        description="When the run finished; timezone-aware, UTC by default.",
    )
    settings: dict[str, str] = Field(
        description="Settings of the run as Settings.as_env() returns them; a Settings "
        "instance is converted. LLM_PROVIDER, for example, tells a fake-mode run from an "
        "Ollama run."
    )

    @field_validator("settings", mode="before")
    @classmethod
    def _snapshot_settings(cls, value: Any) -> Any:
        """Accept a Settings instance and keep its environment-variable form."""
        return value.as_env() if isinstance(value, Settings) else value
