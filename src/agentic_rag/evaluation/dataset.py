"""Evaluation dataset: the question record, the JSON Lines loader and the default path.

The question set is a UTF-8 JSON Lines file, ``data/eval/questions.jsonl`` by default
(:data:`DEFAULT_DATASET_PATH`): one JSON object per line, each one :class:`EvalItem`. Blank
lines are skipped, and a UTF-8 byte order mark at the start of the file is accepted, because
some Windows editors write one. ``data/eval/README.md`` documents the schema with an example
line.

:func:`load_dataset` validates the whole file before it returns anything. A problem is
reported as a :class:`DatasetError` that names the file and the 1-based line number, so a
hand-written file is quick to fix. The committed set has 17 questions over the frontend
documentation corpus (plan decision 8).

Relative paths resolve against the current working directory, like the paths in
``Settings``: the repository root when running locally, ``/app`` in the container.
"""

import codecs
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any, Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from agentic_rag.agent.types import Intent

__all__ = [
    "DEFAULT_DATASET_PATH",
    "DatasetError",
    "EvalItem",
    "load_dataset",
]

logger = logging.getLogger(__name__)

DEFAULT_DATASET_PATH: Final = Path("data/eval/questions.jsonl")
"""Question set of ``agentic-rag eval`` when ``--dataset`` is not given."""

_NonEmptyStr = Annotated[str, Field(min_length=1)]


class EvalItem(BaseModel):
    """One evaluation question with its reference answer and the expected behaviour.

    Items are immutable values. Unknown keys are rejected, so a misspelt key in the dataset
    (``expected_document`` instead of ``expected_documents``) fails loudly instead of being
    ignored. Leading and trailing whitespace is stripped from every string.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    id: str = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
        description="Unique identifier such as 'q01': letters, digits, '.', '_' and '-', "
        "starting with a letter or a digit, so it is safe in file names, tables and logs.",
    )
    question: str = Field(min_length=1, description="The question as a user would ask it.")
    reference_answer: str = Field(
        min_length=1,
        description="An answer a domain expert accepts as correct; answer correctness is "
        "judged against it.",
    )
    expected_documents: list[_NonEmptyStr] = Field(
        default_factory=list,
        description="The documents that contain the answer, each identified as "
        "DocumentMetadata.source stores it (and Source.source repeats it): the path "
        "relative to DATA_DIR with forward slashes, e.g. 'guides/setup.pdf', or a URL. "
        "Empty when the question needs no retrieval. Retrieval hit@k compares them exactly "
        "with the retrieved documents.",
    )
    expected_intent: Intent | None = Field(
        default=None,
        description="The route analyze_request should choose: direct, single, complex or "
        "tool. None leaves routing unchecked for this item. Routing accuracy is computed "
        "against it.",
    )
    tags: list[_NonEmptyStr] = Field(
        default_factory=list,
        description="Free labels for grouping the results, e.g. 'multi-part' or 'out-of-scope'.",
    )
    notes: str | None = Field(
        default=None, description="Free text for reviewers, e.g. why the item is in the set."
    )

    @field_validator("expected_documents", "tags")
    @classmethod
    def _reject_duplicates(cls, values: list[str]) -> list[str]:
        """Reject repeated entries, which are always a mistake in a hand-written dataset."""
        repeated = sorted({value for value in values if values.count(value) > 1})
        if repeated:
            raise ValueError(f"contains duplicates: {', '.join(map(repr, repeated))}")
        return values


class DatasetError(ValueError):
    """An evaluation dataset that cannot be loaded, with the location of the problem.

    ``str(error)`` reads ``<path>:<line>: <reason>``, a format that editors and CI logs turn
    into a link to the offending line. The underlying error, when there is one (text that is
    not UTF-8, invalid JSON, a failed item validation), is chained as ``__cause__``.

    Attributes:
        path: The dataset file.
        line: 1-based number of the offending line.
        reason: What is wrong with that line.
    """

    def __init__(self, path: Path, line: int, reason: str) -> None:
        """Create the error.

        Args:
            path: The dataset file.
            line: 1-based number of the offending line.
            reason: What is wrong with that line.
        """
        super().__init__(path, line, reason)
        self.path = path
        self.line = line
        self.reason = reason

    def __str__(self) -> str:
        """Return the location and the reason as ``<path>:<line>: <reason>``."""
        return f"{self.path}:{self.line}: {self.reason}"


def load_dataset(path: Path) -> list[EvalItem]:
    """Load and validate an evaluation dataset in JSON Lines format.

    Every non-blank line must be a JSON object that validates as an :class:`EvalItem`, with
    no repeated keys and an id that no earlier line used. The whole file is checked before
    anything is returned.

    Args:
        path: The JSON Lines file, usually :data:`DEFAULT_DATASET_PATH`.

    Returns:
        The items in file order; an empty list when the file has no non-blank line.

    Raises:
        FileNotFoundError: If ``path`` does not exist (other ``OSError`` subclasses for other
            read failures).
        DatasetError: For the first invalid line: text that is not UTF-8, invalid JSON, a
            repeated key, a value that does not validate as an item (a value that is not an
            object included), or an id already used on an earlier line.
    """
    items: list[EvalItem] = []
    first_line_of_id: dict[str, int] = {}
    for number, line in _numbered_lines(path):
        if not line.strip():
            continue
        item = _parse_item(line, path, number)
        if item.id in first_line_of_id:
            reason = f"duplicate id {item.id!r}, already used on line {first_line_of_id[item.id]}"
            raise DatasetError(path, number, reason)
        first_line_of_id[item.id] = number
        items.append(item)
    logger.debug("Loaded %d evaluation items from %s", len(items), path)
    return items


def _numbered_lines(path: Path) -> Iterator[tuple[int, str]]:
    r"""Yield the 1-based number and the decoded text of every line of a UTF-8 file.

    The bytes are split on ``\n``, ``\r\n`` and ``\r`` only, before decoding, so
    characters such as U+2028 inside JSON strings do not break a line, and a byte sequence
    that is not UTF-8 is reported with its line number.
    """
    data = path.read_bytes().removeprefix(codecs.BOM_UTF8)
    for number, raw in enumerate(data.splitlines(), start=1):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            reason = f"not valid UTF-8 at byte {exc.start + 1} of the line; save the file as UTF-8"
            raise DatasetError(path, number, reason) from exc
        yield number, text


def _parse_item(line: str, path: Path, number: int) -> EvalItem:
    """Parse one non-blank line into an item, or raise a DatasetError for that line."""
    try:
        data = json.loads(line, object_pairs_hook=_unique_keys)
    except _DuplicateKeyError as exc:
        raise DatasetError(path, number, f"duplicate key {exc.key!r}") from exc
    except json.JSONDecodeError as exc:
        raise DatasetError(path, number, f"invalid JSON: {exc.msg} (column {exc.colno})") from exc
    try:
        return EvalItem.model_validate(data)
    except ValidationError as exc:
        raise DatasetError(path, number, _describe_validation_error(exc)) from exc


class _DuplicateKeyError(ValueError):
    """A key that occurs twice in one JSON object (``json`` alone keeps the last silently)."""

    def __init__(self, key: str) -> None:
        super().__init__(key)
        self.key = key


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """``object_pairs_hook`` for ``json.loads`` that rejects repeated keys."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(key)
        result[key] = value
    return result


def _describe_validation_error(error: ValidationError) -> str:
    """Summarize the problems of a failed item validation on one line."""
    problems = []
    for issue in error.errors(include_url=False):
        location = ".".join(str(part) for part in issue["loc"])
        problems.append(f"{location}: {issue['msg']}" if location else issue["msg"])
    return "invalid item: " + "; ".join(problems)
