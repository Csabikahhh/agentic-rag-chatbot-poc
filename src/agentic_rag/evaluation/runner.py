"""Evaluation runner: the report models, the node targets and ``run_evaluation``.

``run_evaluation`` is the implementation behind ``agentic-rag eval`` (plan section 5.6):

1. Load the question set with ``agentic_rag.evaluation.dataset.load_dataset`` (default
   ``data/eval/questions.jsonl``).
2. Run every question on a fresh state through the full main graph (``target="graph"``) or
   through one node of :data:`NODE_TARGETS` (``target="node"``), with the provider that
   ``Settings.llm_provider`` selects. A question whose run fails records its error, and the
   run goes on with the next one; only a missing or mismatched index stops the evaluation,
   because it would fail every question.
3. Score every outcome with ``agentic_rag.evaluation.metrics``: routing accuracy, retrieval
   hit@k with ``k = Settings.top_k``, and on the full graph answer correctness and
   faithfulness, judged by the Ollama model (``OLLAMA_MODEL``, or ``judge_model``). The fake
   LLM provider cannot judge, so its runs leave the judged metrics out.
4. Write the :class:`EvalReport` as JSON and its Markdown summary (:func:`render_summary`) to
   the output directory (default ``agentic_rag.reports.RESULTS_DIR``, ``data/eval/results/``)
   with :func:`write_report`. ``docs/evaluation.md`` discusses the committed run.

The report models define the format of the committed result files. :class:`EvalItemResult`
rejects verdicts that contradict its own item, :class:`EvalReport` computes its aggregate
scores from its items, so the two never disagree, and the records reject NaN and infinite
numbers, so every report serializes to valid JSON and loads back unchanged.

Importing the module loads neither LangGraph nor a model library: ``run_evaluation`` imports
the agent when it runs, so loading a committed report stays light.
"""

import itertools
import logging
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any, Final, Literal, Self, get_args

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from agentic_rag.agent.types import Intent
from agentic_rag.config import Settings
from agentic_rag.errors import ConfigurationError, IndexNotFoundError, InvalidArgumentError
from agentic_rag.evaluation.dataset import DEFAULT_DATASET_PATH, EvalItem, EvalMessage, load_dataset
from agentic_rag.evaluation.metrics import (
    JudgeError,
    JudgeScore,
    answer_correctness,
    complete_evidence_at_k,
    faithfulness,
    hit_at_k,
    intent_matches,
    mean_score,
)
from agentic_rag.reports import RESULTS_DIR, RunReport

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

__all__ = [
    "NODE_TARGETS",
    "EvalItemResult",
    "EvalReport",
    "EvalScores",
    "EvalTarget",
    "MetricSummary",
    "render_summary",
    "run_evaluation",
    "write_report",
]

logger = logging.getLogger(__name__)

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
    complete_evidence_at_k: bool | None = Field(default=None)
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
        if self.complete_evidence_at_k is not None:
            if not self.item.expected_document_groups:
                raise ValueError("complete_evidence_at_k requires expected_document_groups")
            found = set(itertools.chain(*self.retrieved_documents))
            if self.complete_evidence_at_k and any(
                found.isdisjoint(group) for group in self.item.expected_document_groups
            ):
                raise ValueError("complete_evidence_at_k is True but an evidence group is missing")
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
    complete_evidence_rate_at_k: MetricSummary = Field(
        default_factory=lambda: MetricSummary(mean=None, count=0)
    )

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
    judge_model: str | None = Field(
        default=None,
        description="The Ollama model that judged answer correctness and faithfulness; None "
        "when nothing was judged (the fake LLM provider, or a node target).",
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
            complete_evidence_rate_at_k=_summarize(
                result.complete_evidence_at_k for result in self.items
            ),
            answer_correctness=_summarize(
                _score_of(result.answer_correctness) for result in self.items
            ),
            faithfulness=_summarize(_score_of(result.faithfulness) for result in self.items),
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
    judge_model: str | None = None,
) -> EvalReport:
    """Run the question set through the graph or one node, score it and write the report.

    Every question runs on a fresh state (see the module docstring for the steps). A question
    whose run fails records its error and scores as a failure: no answer, no route, no hit.

    Args:
        settings: The effective settings. The provider, the models and ``top_k`` shape the
            run, and the report records them.
        target: ``"graph"`` runs the full main graph from ``AgentInput``; ``"node"`` runs only
            the node named by ``node``.
        node: The node to evaluate, one of :data:`NODE_TARGETS`; required with
            ``target="node"`` and not allowed with ``target="graph"``.
        dataset_path: The question set; ``None`` means
            ``agentic_rag.evaluation.dataset.DEFAULT_DATASET_PATH``.
        output_dir: Directory for the report files; ``None`` means
            ``agentic_rag.reports.RESULTS_DIR``.
        judge_model: The Ollama model that judges the answers; ``None`` means
            ``settings.ollama_model``. A judge other than the evaluated model keeps a
            comparison of models fair. Ignored by node targets and with the fake LLM
            provider, which judge nothing.

    Returns:
        The report, as written to ``output_dir``.

    Raises:
        InvalidArgumentError: If ``target`` is neither ``"graph"`` nor ``"node"``, if
            ``node`` is missing with ``target="node"`` or given with ``target="graph"``, or if
            it is not one of :data:`NODE_TARGETS`; or if the question set has no question.
        FileNotFoundError: If the question set does not exist.
        DatasetError: If a line of the question set is invalid.
        IndexNotFoundError: If the vector index has not been built.
        ConfigurationError: If the index was built with other embeddings
            (``EmbeddingMismatchError``).
    """
    problem = _target_problem(target, node)
    if problem is not None:
        raise InvalidArgumentError(problem)
    path = dataset_path or DEFAULT_DATASET_PATH
    items = load_dataset(path)
    if not items:
        raise InvalidArgumentError(f"the question set {path} has no question")

    run = _graph_runner(settings) if target == "graph" else _node_runner(settings, node or "")
    judge: BaseChatModel | None = None
    judge_name: str | None = None
    if target == "graph" and settings.llm_provider != "fake":
        judge_name = judge_model or settings.ollama_model
        judge = _judge_model(settings, judge_name)
    logger.info(
        "Evaluating %d questions of %s on the %s with LLM_PROVIDER=%s%s",
        len(items),
        path,
        "full graph" if target == "graph" else f"node {node}",
        settings.llm_provider,
        f" and the judge {judge_name}" if judge_name else "",
    )
    results = []
    for number, item in enumerate(items, start=1):
        result = _evaluate(item, run, judge, target=target, node=node, k=settings.top_k)
        outcome = f"failed: {result.error}" if result.error else "done"
        logger.info("%d/%d %s: %s", number, len(items), item.id, outcome)
        results.append(result)
    report = EvalReport(
        settings=settings,
        target=target,
        node=node,
        dataset=path,
        judge_model=judge_name,
        items=results,
    )
    write_report(report, output_dir or RESULTS_DIR)
    return report


def write_report(report: EvalReport, output_dir: Path) -> Path:
    """Write a report as JSON and its Markdown summary next to it.

    The file name holds the kind of run and the UTC time it finished, so runs never overwrite
    each other: ``eval-graph-20261003T120000Z.json`` or
    ``eval-node-analyze_request-20261003T120000Z.json``, and the same name with ``.md`` for
    :func:`render_summary`.

    Args:
        report: The report to write.
        output_dir: The directory; created when missing.

    Returns:
        The path of the JSON file.
    """
    stamp = report.created_at.strftime("%Y%m%dT%H%M%SZ")
    kind = report.target if report.node is None else f"{report.target}-{report.node}"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"eval-{kind}-{stamp}.json"
    json_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
    summary_path = json_path.with_suffix(".md")
    summary_path.write_text(render_summary(report), encoding="utf-8", newline="\n")
    logger.info("Wrote %s and %s", json_path, summary_path)
    return json_path


def render_summary(report: EvalReport) -> str:
    """Summarize a report as Markdown: the configuration, the scores, every question.

    Sections: the run and its settings; the aggregate scores with the number of questions
    each metric applies to; the scores by tag (for tags of at least two questions, such as
    ``english`` and ``hungarian``); one row per question; the judge's rationales; the errors.

    Args:
        report: The report to summarize.

    Returns:
        The Markdown text, ending with a newline.
    """
    settings = report.settings
    what = "the full graph" if report.target == "graph" else f"the node `{report.node}`"
    llm = settings.get("LLM_PROVIDER", "?")
    if llm == "ollama":
        llm += f" `{settings.get('OLLAMA_MODEL', '?')}`"
    embeddings = settings.get("EMBEDDING_PROVIDER", "?")
    if embeddings == "huggingface":
        embeddings += f" `{settings.get('EMBEDDING_MODEL', '?')}`"
    judge = f"`{report.judge_model}`" if report.judge_model else "none"
    k = settings.get("TOP_K", "k")
    lines = [
        f"# Evaluation of {what}",
        "",
        f"- Run: {report.created_at:%Y-%m-%d %H:%M} UTC, dataset `{report.dataset}`, "
        f"{len(report.items)} questions, {report.error_count} failed",
        f"- LLM: {llm}; judge: {judge}",
        f"- Embeddings: {embeddings}; TOP_K={k}; "
        f"GRADE_WITH_LLM={settings.get('GRADE_WITH_LLM', '?')}",
        "",
        "| Metric | Score | Questions |",
        "|---|---|---|",
    ]
    scores = report.scores
    for label, summary in (
        ("Routing accuracy", scores.routing_accuracy),
        (f"Retrieval hit@{k}", scores.hit_rate_at_k),
        (f"Complete evidence@{k}", scores.complete_evidence_rate_at_k),
        ("Answer correctness", scores.answer_correctness),
        ("Faithfulness", scores.faithfulness),
    ):
        lines.append(f"| {label} | {_format_score(summary.mean)} | {summary.count} |")
    latencies = [result.latency_ms for result in report.items if result.latency_ms is not None]
    if latencies:
        lines += [
            "",
            f"Latency per question: mean {_format_seconds(mean_score(latencies) or 0.0)}, "
            f"max {_format_seconds(max(latencies))}.",
        ]
    lines += _tag_table(report.items, k)
    lines += [
        "",
        "## Questions",
        "",
        f"| Id | Tags | Route (expected → chosen) | hit@{k} | Correctness | Faithfulness "
        "| Latency |",
        "|---|---|---|---|---|---|---|",
    ]
    for result in report.items:
        route = f"{result.item.expected_intent or '–'} → {result.predicted_intent or '–'}"
        if result.intent_correct is not None:
            route += f" {_mark(result.intent_correct)}"
        lines.append(
            f"| {result.item.id} | {', '.join(result.item.tags)} | {route} | "
            f"{_mark(result.hit_at_k)} | "
            f"{_format_judged(result.answer_correctness)} | "
            f"{_format_judged(result.faithfulness)} | "
            f"{_format_seconds(result.latency_ms) if result.latency_ms is not None else '–'} |"
        )
    rationales = [
        f"- **{result.item.id}**: {_rationales(result)}"
        for result in report.items
        if result.answer_correctness is not None or result.faithfulness is not None
    ]
    if rationales:
        lines += ["", "## Judge rationales", "", *rationales]
    errors = [f"- **{r.item.id}**: {r.error}" for r in report.items if r.error is not None]
    evidence = [r for r in report.items if r.complete_evidence_at_k is not None]
    if evidence:
        lines += [
            "",
            "## Complete evidence",
            "",
            f"Every required evidence group must have a hit in a sub-task's top {k}.",
            "",
        ]
        lines += [f"- **{r.item.id}**: {_mark(r.complete_evidence_at_k)}" for r in evidence]
    if errors:
        lines += ["", "## Errors", "", *errors]
    return "\n".join(lines) + "\n"


@dataclass
class _Outcome:
    """What one run of a question produced, before scoring."""

    answer: str | None = None
    intent: Intent | None = None
    retrieved: list[list[str]] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)
    grounded: bool = False  # The run searched or called a tool, whatever that produced.


type _Runner = Callable[..., _Outcome]
"""Runs one question and returns its outcome."""


def _graph_runner(settings: Settings) -> _Runner:
    """Build the main graph once and return a runner of single questions through it."""
    from langchain_core.messages import HumanMessage

    from agentic_rag.agent.graph import build_agent_graph

    graph = build_agent_graph(settings)

    def run(question: str, *, history: Sequence[EvalMessage] = ()) -> _Outcome:
        output = graph.invoke({"messages": [*_history_messages(history), HumanMessage(question)]})
        results = output.get("subtask_results", [])
        return _Outcome(
            answer=output.get("answer"),
            intent=output.get("intent"),
            retrieved=[_documents(result) for result in results if result.kind == "retrieve"],
            contexts=[result.output for result in results if result.ok and result.output.strip()],
            grounded=bool(results),
        )

    return run


def _node_runner(settings: Settings, node: str) -> _Runner:
    """Return a runner that drives one node of :data:`NODE_TARGETS` with its dependencies."""
    from langchain_core.messages import HumanMessage

    from agentic_rag.agent import nodes
    from agentic_rag.agent.state import Subtask
    from agentic_rag.agent.tools import get_tools
    from agentic_rag.llm import get_chat_model

    search_tool, *other_tools = get_tools(settings)
    if node == "analyze_request":
        chat_model = get_chat_model(settings)
        tools = {tool.name: tool for tool in other_tools}

        def analyze(question: str, *, history: Sequence[EvalMessage] = ()) -> _Outcome:
            state = {"messages": [*_history_messages(history), HumanMessage(question)]}
            update = nodes.analyze_request(state, chat_model=chat_model, tools=tools)  # type: ignore[arg-type]
            return _Outcome(answer=update.get("draft_answer"), intent=update.get("intent"))

        return analyze

    def retrieve(question: str, *, history: Sequence[EvalMessage] = ()) -> _Outcome:
        if history:
            raise InvalidArgumentError(
                "Follow-up retrieval must be evaluated through the full graph"
            )
        subtask = Subtask(id="s1", kind="retrieve", input=question)
        update = nodes.run_rag_subtask(
            {"subtask": subtask, "question": question},
            search_tool=search_tool,  # type: ignore[arg-type]
        )
        (result,) = update["subtask_results"]
        return _Outcome(answer=result.output or None, retrieved=[_documents(result)])

    return retrieve


def _history_messages(history: Sequence[EvalMessage]) -> list[Any]:
    """Convert fixed evaluation history into the same message types as the chat UI."""
    from langchain_core.messages import AIMessage, HumanMessage

    return [(HumanMessage if turn.role == "user" else AIMessage)(turn.content) for turn in history]


def _judge_model(settings: Settings, model: str) -> "BaseChatModel":
    """The chat model that judges: the Ollama provider of the settings with ``model``."""
    from agentic_rag.llm import get_chat_model

    return get_chat_model(Settings.model_validate({**settings.model_dump(), "ollama_model": model}))


def _documents(result: Any) -> list[str]:
    """The ranked documents of a retrieve sub-task result (``Source.source`` of each chunk)."""
    return [source.source for source in result.sources]


def _evaluate(
    item: EvalItem,
    run: _Runner,
    judge: "BaseChatModel | None",
    *,
    target: EvalTarget,
    node: str | None,
    k: int,
) -> EvalItemResult:
    """Run one question, score its outcome and return its result."""
    started = time.perf_counter()
    error: str | None = None
    try:
        outcome = run(item.question, history=item.history) if item.history else run(item.question)
    except (IndexNotFoundError, ConfigurationError):
        raise  # Not specific to this question: every other one would fail the same way.
    except Exception as exc:
        logger.warning("Question %s failed", item.id, exc_info=True)
        outcome, error = _Outcome(), f"{type(exc).__name__}: {exc}"
    latency_ms = (time.perf_counter() - started) * 1000.0

    routes = target == "graph" or node == "analyze_request"
    retrieves = target == "graph" or node == "run_rag_subtask"
    correctness = support = None
    if judge is not None and outcome.answer:
        correctness = _judged(
            item,
            "answer correctness",
            lambda: answer_correctness(
                "\n".join([*(f"{m.role}: {m.content}" for m in item.history), item.question]),
                outcome.answer or "",
                item.reference_answer,
                judge=judge,
            ),
        )
        if outcome.grounded:
            support = _judged(
                item,
                "faithfulness",
                lambda: faithfulness(outcome.answer or "", outcome.contexts, judge=judge),
            )
    return EvalItemResult(
        item=item,
        answer=outcome.answer,
        predicted_intent=outcome.intent,
        retrieved_documents=outcome.retrieved,
        intent_correct=intent_matches(item.expected_intent, outcome.intent) if routes else None,
        hit_at_k=hit_at_k(outcome.retrieved, item.expected_documents, k) if retrieves else None,
        complete_evidence_at_k=(
            complete_evidence_at_k(outcome.retrieved, item.expected_document_groups, k)
            if retrieves
            else None
        ),
        answer_correctness=correctness,
        faithfulness=support,
        latency_ms=latency_ms,
        error=error,
    )


def _judged(item: EvalItem, metric: str, judge: Callable[[], JudgeScore]) -> JudgeScore | None:
    """Run one judge call; a reply that cannot be read leaves the metric out, with a warning."""
    try:
        return judge()
    except JudgeError as exc:
        logger.warning("Question %s: %s not judged: %s", item.id, metric, exc)
        return None


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


def _score_of(score: JudgeScore | None) -> float | None:
    """Return the numeric score of an LLM-judged metric, or None when it was not judged."""
    return None if score is None else score.score


def _tag_table(results: Sequence[EvalItemResult], k: str) -> list[str]:
    """The scores by tag, for the tags of at least two questions; empty without such tags."""
    by_tag: dict[str, list[EvalItemResult]] = defaultdict(list)
    for result in results:
        for tag in result.item.tags:
            by_tag[tag].append(result)
    rows = [
        f"| {tag} | {len(group)} | "
        f"{_format_score(mean_score(r.intent_correct for r in group))} | "
        f"{_format_score(mean_score(r.hit_at_k for r in group))} | "
        f"{_format_score(mean_score(_score_of(r.answer_correctness) for r in group))} | "
        f"{_format_score(mean_score(_score_of(r.faithfulness) for r in group))} |"
        for tag, group in sorted(by_tag.items())
        if len(group) >= 2
    ]
    if not rows:
        return []
    return [
        "",
        "## By tag",
        "",
        f"| Tag | Questions | Routing | hit@{k} | Correctness | Faithfulness |",
        "|---|---|---|---|---|---|",
        *rows,
    ]


def _format_score(score: float | None) -> str:
    """A mean score with two decimals, or a dash when the metric applied to no question."""
    return "–" if score is None else f"{score:.2f}"


def _format_judged(score: JudgeScore | None) -> str:
    """A judged score with one decimal, or a dash when it was not judged."""
    return "–" if score is None else f"{score.score:.1f}"


def _format_seconds(milliseconds: float) -> str:
    """A latency in seconds with one decimal."""
    return f"{milliseconds / 1000.0:.1f} s"


def _mark(verdict: bool | None) -> str:
    """A check mark for a passed check, a cross for a failed one, a dash for none."""
    if verdict is None:
        return "–"
    return "✓" if verdict else "✗"


def _rationales(result: EvalItemResult) -> str:
    """The judge's rationales of one question, on one line."""
    parts = []
    if result.answer_correctness is not None:
        parts.append(
            f"correctness {result.answer_correctness.score:.1f}: "
            f"{result.answer_correctness.rationale}"
        )
    if result.faithfulness is not None:
        parts.append(
            f"faithfulness {result.faithfulness.score:.1f}: {result.faithfulness.rationale}"
        )
    return " · ".join(parts)
