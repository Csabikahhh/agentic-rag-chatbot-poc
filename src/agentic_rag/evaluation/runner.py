"""Evaluation runner: the report models, the node targets and ``run_evaluation``.

``run_evaluation`` (Phase 7) is the implementation behind ``agentic-rag eval`` (plan section
5.6):

1. Load the question set with ``agentic_rag.evaluation.dataset.load_dataset`` (default
   ``data/eval/questions.jsonl``).
2. Run every question on a fresh state through the full main graph (``target="graph"``) or
   through one node of :data:`NODE_TARGETS` (``target="node"``), with the provider that
   ``Settings.llm_provider`` selects.
3. Score every outcome with ``agentic_rag.evaluation.metrics``: routing accuracy, retrieval
   hit@k with ``k = Settings.top_k``, answer correctness and faithfulness.
4. Write the :class:`EvalReport` as JSON to the output directory (default
   ``agentic_rag.reports.RESULTS_DIR``, ``data/eval/results/``); the written summary goes to
   ``docs/evaluation.md``.

The report models are real already: they define the format of the committed result files.
:class:`EvalItemResult` rejects verdicts that contradict its own item, :class:`EvalReport`
computes its aggregate scores from its items, so the two never disagree, and the records
reject NaN and infinite numbers, so every report serializes to valid JSON and loads back
unchanged.
"""

import itertools
from collections.abc import Iterable
from pathlib import Path, PurePath
from typing import Any, Final, Literal, Self, get_args

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from agentic_rag.agent.types import Intent
from agentic_rag.config import Settings
from agentic_rag.errors import InvalidArgumentError, planned
from agentic_rag.evaluation.dataset import EvalItem
from agentic_rag.evaluation.metrics import JudgeScore, intent_matches, mean_score
from agentic_rag.reports import RunReport

__all__ = [
    "NODE_TARGETS",
    "EvalItemResult",
    "EvalReport",
    "EvalScores",
    "EvalTarget",
    "MetricSummary",
    "run_evaluation",
]

EvalTarget = Literal["graph", "node"]
"""What ``run_evaluation`` runs: the full main graph, or one node of :data:`NODE_TARGETS`."""

NODE_TARGETS: Final[tuple[str, ...]] = ("analyze_request", "run_rag_subtask")
"""The main-graph nodes that ``target="node"`` evaluates: an ``EvalItem`` alone provides their
input, and a metric scores their output.

- ``analyze_request`` runs on the question as the only user message, the input of a graph
  run. Its ``intent`` gives routing accuracy against ``expected_intent`` for one LLM call per
  question, without running (or failing in) the later nodes. It retrieves nothing, so hit@k
  is not computed.
- ``run_rag_subtask`` runs on one ``retrieve`` sub-task whose query is the question, like the
  one-step plan that ``analyze_request`` writes on the ``single`` route. The ranked documents
  of its ``SubtaskResult`` give hit@k against ``expected_documents`` with the router out of
  the way, so a misrouted question still measures retrieval; for a multi-part question it is
  a one-shot baseline for the decomposition. It returns a context, not an answer, so the
  routing and judged metrics are not computed.

The other nodes are evaluated through the full graph. ``plan_subtasks`` could run from the
question, but it returns a plan, which no metric scores. ``call_tool`` needs a tool sub-task
with its arguments, and ``synthesize_answer``, ``verify_answer`` and ``finalize_response``
need sub-task results or a draft answer; an item carries none of them.
"""

_TARGETS: Final[tuple[str, ...]] = get_args(EvalTarget)


class EvalItemResult(BaseModel):
    """Outcome and scores of one evaluation question.

    A metric field is ``None`` when the metric does not apply to the item or is not computed
    for the evaluated target (for example hit@k when only ``analyze_request`` runs). A verdict
    that is set must agree with the item: ``intent_correct`` is
    ``metrics.intent_matches(item.expected_intent, predicted_intent)``, and ``hit_at_k`` needs
    expected documents and, when it is ``True``, one of them among ``retrieved_documents``.
    Results are immutable values.
    """

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    item: EvalItem = Field(description="The question with its reference answer and expectations.")
    answer: str | None = Field(
        default=None,
        description="The final answer, or the output text of the evaluated node; None when "
        "the run produced none.",
    )
    predicted_intent: Intent | None = Field(
        default=None, description="The intent analyze_request chose; None when it did not run."
    )
    retrieved_documents: list[list[str]] = Field(
        default_factory=list,
        description="One list per retrieve sub-task of the run: the documents of the "
        "sub-task's chunks (SubtaskResult.sources) in rank order, best first, each the path "
        "relative to DATA_DIR with forward slashes (DocumentMetadata.source, as Source.source "
        "repeats it) or a URL. The input of hit@k, which is computed per retrieve sub-task; "
        "never taken from AgentOutput.sources. Empty when no retrieve sub-task ran.",
    )
    intent_correct: bool | None = Field(
        default=None,
        description="Routing check of this item: metrics.intent_matches(item.expected_intent, "
        "predicted_intent); None when the item has no expected intent or routing is not "
        "evaluated for the target.",
    )
    hit_at_k: bool | None = Field(
        default=None,
        description="Retrieval check of this item: metrics.hit_at_k(retrieved_documents, "
        "item.expected_documents, k) with k = TOP_K; None when the item has no expected "
        "documents or retrieval is not evaluated for the target.",
    )
    answer_correctness: JudgeScore | None = Field(
        default=None, description="Judged agreement with the reference answer."
    )
    faithfulness: JudgeScore | None = Field(
        default=None, description="Judged support of the answer by its contexts."
    )
    latency_ms: float | None = Field(
        default=None, ge=0, description="Wall-clock time of the run for this item."
    )
    error: str | None = Field(
        default=None, description="Why the run failed; None when it succeeded."
    )

    @model_validator(mode="after")
    def _check_verdicts(self) -> Self:
        """Reject verdicts that contradict the item or the recorded outcome.

        ``None`` stays legal wherever a metric was not computed. ``k`` is not part of the
        result, so a ``hit_at_k`` of ``True`` is only checked for a retrieved expected
        document at any rank.
        """
        expected_intent = self.item.expected_intent
        routed_right = intent_matches(expected_intent, self.predicted_intent)
        if self.intent_correct is not None and self.intent_correct != routed_right:
            msg = (
                f"intent_correct={self.intent_correct} contradicts the item: "
                f"expected_intent={expected_intent!r}, predicted_intent={self.predicted_intent!r}"
            )
            raise ValueError(msg)
        if self.hit_at_k is None:
            return self
        expected = set(self.item.expected_documents)
        if not expected:
            raise ValueError("hit_at_k must be None when the item has no expected_documents")
        if self.hit_at_k and expected.isdisjoint(itertools.chain(*self.retrieved_documents)):
            raise ValueError("hit_at_k is True, but no expected document was retrieved")
        return self


class MetricSummary(BaseModel):
    """Aggregate of one metric over the items it applies to.

    ``mean`` is ``None`` exactly when ``count`` is 0, i.e. when the metric applied to no item.
    Summaries are immutable values.
    """

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    mean: float | None = Field(
        ge=0.0, le=1.0, description="Mean score over the items the metric applies to."
    )
    count: int = Field(ge=0, description="Number of items the metric applies to.")

    @model_validator(mode="after")
    def _check_count(self) -> Self:
        """Keep ``mean`` and ``count`` consistent."""
        if (self.count == 0) != (self.mean is None):
            raise ValueError("mean must be None exactly when count is 0")
        return self


class EvalScores(BaseModel):
    """Aggregate scores of one evaluation run, one per metric of plan section 5.6.

    Scores are immutable values.
    """

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    routing_accuracy: MetricSummary = Field(
        description="Share of the items with an expected intent that were routed to it."
    )
    hit_rate_at_k: MetricSummary = Field(
        description="Share of the items with expected documents that retrieved one in the top "
        "k of a retrieve sub-task."
    )
    answer_correctness: MetricSummary = Field(
        description="Mean judged agreement of the answers with the reference answers."
    )
    faithfulness: MetricSummary = Field(
        description="Mean judged support of the answers by their contexts."
    )


class EvalReport(RunReport):
    """Result of one evaluation run: the format of the evaluation files in ``data/eval/results/``.

    ``created_at`` and the ``settings`` snapshot come from :class:`~agentic_rag.reports.RunReport`.
    ``scores`` and ``error_count`` are computed from ``items``. They appear in the JSON output,
    and loading a report back recomputes them instead of reading them. Reports are immutable
    values.
    """

    target: EvalTarget = Field(
        description="'graph' for the full main graph, 'node' for one node of NODE_TARGETS."
    )
    node: str | None = Field(
        default=None,
        description="The evaluated node, one of NODE_TARGETS; set exactly when target is 'node'.",
    )
    dataset: str = Field(
        min_length=1,
        description="Path of the question set, with forward slashes; a Path is converted.",
    )
    items: list[EvalItemResult] = Field(description="One result per question, in dataset order.")

    @field_validator("dataset", mode="before")
    @classmethod
    def _path_as_posix(cls, value: Any) -> Any:
        """Store paths with forward slashes, so result files look the same on every OS."""
        return value.as_posix() if isinstance(value, PurePath) else value

    @model_validator(mode="after")
    def _check_node(self) -> Self:
        """Accept only the target and node combinations that ``run_evaluation`` runs."""
        problem = _target_problem(self.target, self.node)
        if problem is not None:
            raise ValueError(problem)
        return self

    @computed_field
    @property
    def scores(self) -> EvalScores:
        """Aggregate scores over the items; each metric averages the items it applies to."""
        return EvalScores(
            routing_accuracy=_summarize(result.intent_correct for result in self.items),
            hit_rate_at_k=_summarize(result.hit_at_k for result in self.items),
            answer_correctness=_summarize(
                _judged(result.answer_correctness) for result in self.items
            ),
            faithfulness=_summarize(_judged(result.faithfulness) for result in self.items),
        )

    @computed_field
    @property
    def error_count(self) -> int:
        """Number of items whose run failed."""
        return sum(result.error is not None for result in self.items)


def run_evaluation(
    settings: Settings,
    *,
    target: EvalTarget = "graph",
    node: str | None = None,
    dataset_path: Path | None = None,
    output_dir: Path | None = None,
) -> EvalReport:
    """Run the question set through the graph or one node and score it (planned for Phase 7).

    Planned behaviour (see the module docstring): load the questions, run each one on a fresh
    state, score it with ``agentic_rag.evaluation.metrics``, record a failing item's error
    and carry on with the next one, then write the report as JSON to ``output_dir``. The
    target and the node are checked already, before anything runs.

    Args:
        settings: The effective settings. The provider, the models and ``top_k`` shape the
            run, and the report records them.
        target: ``"graph"`` runs the full main graph from ``AgentInput``; ``"node"`` runs only
            the node named by ``node``.
        node: The node to evaluate, one of :data:`NODE_TARGETS`; required with
            ``target="node"`` and not allowed with ``target="graph"``.
        dataset_path: The question set; ``None`` means
            ``agentic_rag.evaluation.dataset.DEFAULT_DATASET_PATH``.
        output_dir: Directory for the JSON report; ``None`` means
            ``agentic_rag.reports.RESULTS_DIR``.

    Returns:
        The report, as written to ``output_dir``.

    Raises:
        InvalidArgumentError: If ``target`` is neither ``"graph"`` nor ``"node"``, if
            ``node`` is missing with ``target="node"`` or given with ``target="graph"``, or if
            it is not one of :data:`NODE_TARGETS`.
        PlannedFeatureError: For valid arguments, until Phase 7.
    """
    problem = _target_problem(target, node)
    if problem is not None:
        raise InvalidArgumentError(problem)
    raise planned(f"{__name__}.run_evaluation", 7)


def _target_problem(target: str, node: str | None) -> str | None:
    """Return why a target and a node do not go together, or None when they do."""
    if target not in _TARGETS:
        return f"target must be 'graph' or 'node', got {target!r}"
    if target == "graph":
        return None if node is None else "a node can only be given with target 'node'"
    if node is None:
        return "target 'node' needs the name of the node to evaluate"
    if node not in NODE_TARGETS:
        return (
            f"node {node!r} cannot be evaluated on its own; choose one of "
            f"{', '.join(NODE_TARGETS)}, or evaluate the full graph"
        )
    return None


def _summarize(scores: Iterable[float | None]) -> MetricSummary:
    """Aggregate per-item scores, skipping the items a metric does not apply to."""
    applicable = [score for score in scores if score is not None]
    return MetricSummary(mean=mean_score(applicable), count=len(applicable))


def _judged(score: JudgeScore | None) -> float | None:
    """Return the numeric score of an LLM-judged metric, or None when it was not judged."""
    return None if score is None else score.score
