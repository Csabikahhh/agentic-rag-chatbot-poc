"""Evaluation metrics (plan section 5.6).

Deterministic metrics, implemented as pure functions:

- Retrieval hit@k: :func:`hit_at_k` for one item, :func:`hit_rate_at_k` over the set. hit@k
  is computed per retrieve sub-task: an item scores a hit when at least one of its retrieve
  sub-tasks has, among its first ``k`` chunks, a chunk from a document listed in
  ``EvalItem.expected_documents``.
- Routing accuracy: :func:`intent_matches` for one item, :func:`routing_accuracy` over the
  set. An item is correct when ``analyze_request`` chose ``EvalItem.expected_intent``.

Where the retrieved documents come from: every retrieve sub-task of a run has a
``SubtaskResult`` whose ``sources`` are its chunks in rank order, best first, and
``[source.source for source in result.sources]`` is its ranked list of documents. Never use
``AgentOutput.sources`` instead: those are the cited chunks, de-duplicated and merged across
the sub-tasks in citation order, so their positions are not ranks.

Metrics judged by the local LLM (LLM-as-judge), planned for Phase 7:

- Answer correctness (:func:`answer_correctness`): how well the answer agrees with the
  reference answer.
- Faithfulness (:func:`faithfulness`): how well the claims of the answer are supported by the
  retrieved context and the tool outputs.

Both return a :class:`JudgeScore` and receive the judge model as an argument: the runner
passes ``get_chat_model(settings)``, tests pass a ``ScriptedChatModel``. Either provider
supports the structured output the judge prompt needs.

Conventions: scores lie in [0, 1] and higher is better. A per-item function returns ``None``
when the metric does not apply to the item (no expected documents, no expected intent). The
aggregates average the applicable items only and return ``None`` when no item applies (see
:func:`mean_score`), so "not measured" never reads as a score of 0.
"""

import itertools
import math
from collections.abc import Collection, Iterable, Sequence, Sized
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from agentic_rag.agent.types import Intent
from agentic_rag.errors import planned

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

__all__ = [
    "JudgeScore",
    "answer_correctness",
    "faithfulness",
    "hit_at_k",
    "hit_rate_at_k",
    "intent_matches",
    "mean_score",
    "routing_accuracy",
]


class JudgeScore(BaseModel):
    """Verdict of an LLM-judged metric on one item.

    Scores are immutable values.
    """

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    score: float = Field(ge=0.0, le=1.0, description="From 0 (worst) to 1 (best).")
    rationale: str = Field(default="", description="The judge's short justification.")


def hit_at_k(retrieved: Sequence[Sequence[str]], expected: Collection[str], k: int) -> bool | None:
    """Tell whether a retrieve sub-task of an item ranked an expected document in its top ``k``.

    Every retrieve sub-task is ranked on its own, so the best chunk of a second sub-task is
    rank 1 of that sub-task, not rank ``k + 1`` of the item. ``k`` counts chunks, not distinct
    documents: several chunks of one document take several ranks. Identifiers are compared
    exactly, letter case and path separators included, so write ``EvalItem.expected_documents``
    in the form ``Source.source`` uses: the path relative to ``DATA_DIR`` with forward slashes
    (``DocumentMetadata.source``), or a URL.

    Args:
        retrieved: One ranked list per retrieve sub-task of the run: the documents of the
            sub-task's chunks in rank order, best first, i.e.
            ``[source.source for source in result.sources]`` for its ``SubtaskResult``
            (never ``AgentOutput.sources``, see the module docstring). A list may hold fewer
            than ``k`` documents; ``retrieved`` is empty when no retrieve sub-task ran.
        expected: The expected documents, ``EvalItem.expected_documents``.
        k: Number of top-ranked chunks to consider per sub-task, at least 1; usually
            ``Settings.top_k``.

    Returns:
        ``True`` for a hit and ``False`` for a miss, which includes a run without any
        retrieve sub-task; ``None`` when ``expected`` is empty, i.e. the item needs no
        retrieval and the metric does not apply.

    Raises:
        ValueError: If ``k`` is less than 1.
        TypeError: If ``expected`` is a single string, or ``retrieved`` does not hold one list
            of document identifiers per sub-task (for example a flat list of identifiers).
    """
    _check_k(k)
    _check_rankings(retrieved)
    _check_not_text(expected, "expected")
    if not expected:
        return None
    wanted = set(expected)
    return any(
        document in wanted for ranked in retrieved for document in itertools.islice(ranked, k)
    )


def hit_rate_at_k(
    retrieved: Sequence[Sequence[Sequence[str]]], expected: Sequence[Collection[str]], k: int
) -> float | None:
    """Return the share of items with an expected document in the top ``k`` of a sub-task.

    Items without expected documents do not apply and are skipped (see :func:`hit_at_k`).

    Args:
        retrieved: Per item, one ranked list of documents per retrieve sub-task, as
            :func:`hit_at_k` takes them.
        expected: Per item, the expected documents, parallel to ``retrieved``.
        k: Number of top-ranked chunks to consider per sub-task, at least 1.

    Returns:
        The hit rate in [0, 1] over the applicable items; ``None`` when no item applies.

    Raises:
        ValueError: If ``k`` is less than 1 or the two sequences differ in length.
        TypeError: As :func:`hit_at_k`, for input that is not nested as described.
    """
    _check_k(k)
    _check_same_length(retrieved, expected, "retrieved", "expected")
    return mean_score(
        hit_at_k(ranked, wanted, k) for ranked, wanted in zip(retrieved, expected, strict=True)
    )


def intent_matches(expected: Intent | None, predicted: Intent | None) -> bool | None:
    """Tell whether the route chosen by ``analyze_request`` is the expected one.

    Args:
        expected: ``EvalItem.expected_intent``; ``None`` when routing is not checked for the
            item.
        predicted: The intent of the run; ``None`` when the run produced none (for example
            because it failed), which counts as wrong.

    Returns:
        Whether the intents are equal; ``None`` when ``expected`` is ``None``.
    """
    if expected is None:
        return None
    return predicted == expected


def routing_accuracy(
    expected: Sequence[Intent | None], predicted: Sequence[Intent | None]
) -> float | None:
    """Return the share of items routed to their expected intent.

    Items without an expected intent are skipped (see :func:`intent_matches`).

    Args:
        expected: Per item, ``EvalItem.expected_intent``.
        predicted: Per item, the intent of the run, parallel to ``expected``.

    Returns:
        The accuracy in [0, 1] over the applicable items; ``None`` when no item applies.

    Raises:
        ValueError: If the two sequences differ in length.
    """
    _check_same_length(expected, predicted, "expected", "predicted")
    return mean_score(
        intent_matches(wanted, chosen) for wanted, chosen in zip(expected, predicted, strict=True)
    )


def mean_score(scores: Iterable[float | None]) -> float | None:
    """Average the per-item scores of the items a metric applies to.

    Booleans count as 1.0 and 0.0; ``None`` marks an item the metric does not apply to and
    is skipped.

    Args:
        scores: Per-item scores, e.g. :func:`hit_at_k` results or ``JudgeScore.score``.

    Returns:
        The arithmetic mean of the scores that are not ``None``; ``None`` when there are none.
    """
    values = [float(score) for score in scores if score is not None]
    if not values:
        return None
    return math.fsum(values) / len(values)


def answer_correctness(
    question: str, answer: str, reference_answer: str, *, judge: "BaseChatModel"
) -> JudgeScore:
    """Judge how well an answer agrees with the reference answer (planned for Phase 7).

    The judge model reads the question, the reference answer and the system's answer and
    rates agreement in substance (facts, numbers, conclusions) rather than in wording; a
    partially correct answer scores in between.

    Args:
        question: The evaluation question.
        answer: The system's final answer.
        reference_answer: ``EvalItem.reference_answer``.
        judge: The chat model that judges, usually ``get_chat_model(settings)``.

    Returns:
        The score in [0, 1] with the judge's rationale.

    Raises:
        PlannedFeatureError: Always, until Phase 7.
    """
    raise planned(f"{__name__}.answer_correctness", 7)


def faithfulness(answer: str, contexts: Sequence[str], *, judge: "BaseChatModel") -> JudgeScore:
    """Judge whether the claims of an answer are supported by its contexts (Phase 7).

    The judge model checks the answer against the texts it was generated from and penalizes
    claims that the contexts do not support, whether or not they happen to be true. Items
    without any context (``direct`` answers) are not scored.

    Args:
        answer: The system's final answer.
        contexts: The texts the answer may rely on: the RAG contexts and the tool outputs of
            the sub-task results.
        judge: The chat model that judges, usually ``get_chat_model(settings)``.

    Returns:
        The score in [0, 1] with the judge's rationale.

    Raises:
        PlannedFeatureError: Always, until Phase 7.
    """
    raise planned(f"{__name__}.faithfulness", 7)


def _check_k(k: int) -> None:
    """Reject a cut-off that selects no chunk."""
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")


def _check_rankings(retrieved: Sequence[Sequence[str]]) -> None:
    """Reject retrieval input that would silently score as a miss.

    A flat list of identifiers would be read as one ranking per identifier, character by
    character, and ``Source`` records in place of their ``source`` strings would never match.
    """
    _check_not_text(retrieved, "retrieved")
    for ranked in retrieved:
        if isinstance(ranked, str | bytes) or not all(isinstance(doc, str) for doc in ranked):
            msg = (
                "retrieved must hold one list of document identifiers (Source.source) per "
                "retrieve sub-task"
            )
            raise TypeError(msg)


def _check_not_text(value: object, name: str) -> None:
    """Reject a single string where a collection is expected.

    A string is itself a sequence of one-character strings, so it would otherwise be
    compared character by character without any error.
    """
    if isinstance(value, str | bytes):
        raise TypeError(f"{name} must be a collection, not a string")


def _check_same_length(first: Sized, second: Sized, first_name: str, second_name: str) -> None:
    """Reject per-item sequences that are not parallel."""
    if len(first) != len(second):
        msg = (
            f"{first_name} and {second_name} must have one entry per item, "
            f"got {len(first)} and {len(second)}"
        )
        raise ValueError(msg)
