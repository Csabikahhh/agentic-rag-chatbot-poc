"""Tests for agentic_rag.evaluation and agentic_rag.reports: loader, metrics and report models.

Everything runs offline on temporary files and synthetic results. Invalid input is checked by
its 1-based line, its field and Pydantic's stable error type, not by the wording of a library
message. The Phase 7 stubs are checked for their planned-phase error, for the argument checks
that run before it and for the call signature that the CLI uses.
"""

import codecs
import inspect
import json
import pickle
import re
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from agentic_rag.agent.types import Intent
from agentic_rag.config import Settings
from agentic_rag.errors import InvalidArgumentError, PlannedFeatureError, planned
from agentic_rag.evaluation.dataset import (
    DEFAULT_DATASET_PATH,
    DatasetError,
    EvalItem,
    load_dataset,
)
from agentic_rag.evaluation.metrics import (
    JudgeScore,
    answer_correctness,
    faithfulness,
    hit_at_k,
    hit_rate_at_k,
    intent_matches,
    mean_score,
    routing_accuracy,
)
from agentic_rag.evaluation.runner import (
    NODE_TARGETS,
    EvalItemResult,
    EvalReport,
    MetricSummary,
    run_evaluation,
)
from agentic_rag.reports import RESULTS_DIR, RunReport

REPO_ROOT = Path(__file__).resolve().parents[1]
# The main-graph nodes whose input an evaluation item does not carry (see NODE_TARGETS).
NODES_WITHOUT_ITEM_INPUT = (
    "plan_subtasks",
    "call_tool",
    "synthesize_answer",
    "verify_answer",
    "finalize_response",
)

FULL_ITEM: dict[str, Any] = {
    "id": "q01",
    "question": "  What does the first document say?  ",
    "reference_answer": "It says what it says.",
    "expected_documents": ["guides/first.md"],
    "expected_intent": "single",
    "tags": ["single-hop"],
    "notes": "Illustrative only.",
}
MINIMAL_ITEM: dict[str, Any] = {"id": "q02", "question": "Hello?", "reference_answer": "Hi."}


def write_lines(path: Path, *lines: str | dict[str, Any]) -> Path:
    """Write a JSON Lines file: dicts are serialized, strings are written as they are."""
    text = "\n".join(line if isinstance(line, str) else json.dumps(line) for line in lines)
    path.write_text(text + "\n", encoding="utf-8", newline="\n")
    return path


def load_error(tmp_path: Path, line: str | dict[str, Any]) -> DatasetError:
    """Load a dataset whose third line is ``line`` and return the error it raises."""
    path = write_lines(tmp_path / "questions.jsonl", FULL_ITEM, "", line, MINIMAL_ITEM)
    with pytest.raises(DatasetError) as caught:
        load_dataset(path)
    error = caught.value
    assert (error.path, error.line) == (path, 3)
    assert str(error) == f"{path}:3: {error.reason}"
    return error


# --- dataset ---------------------------------------------------------------------------------


def test_load_dataset_returns_the_items_in_file_order_and_skips_blank_lines(
    tmp_path: Path,
) -> None:
    path = write_lines(tmp_path / "questions.jsonl", FULL_ITEM, "", "   \t", MINIMAL_ITEM)

    first, second = load_dataset(path)

    assert first == EvalItem(
        id="q01",
        question="What does the first document say?",
        reference_answer="It says what it says.",
        expected_documents=["guides/first.md"],
        expected_intent="single",
        tags=["single-hop"],
        notes="Illustrative only.",
    )
    assert first.question == "What does the first document say?"
    assert (second.id, second.expected_documents, second.expected_intent) == ("q02", [], None)
    assert (second.tags, second.notes) == ([], None)


def test_load_dataset_accepts_a_byte_order_mark_and_every_line_ending(tmp_path: Path) -> None:
    lines = [
        json.dumps(FULL_ITEM),
        json.dumps(MINIMAL_ITEM),
        json.dumps({**MINIMAL_ITEM, "id": "q3"}),
    ]
    path = tmp_path / "questions.jsonl"
    path.write_bytes(codecs.BOM_UTF8 + f"{lines[0]}\r\n{lines[1]}\r{lines[2]}\n".encode())

    assert [item.id for item in load_dataset(path)] == ["q01", "q02", "q3"]


def test_load_dataset_keeps_non_ascii_text_and_unicode_line_separators(tmp_path: Path) -> None:
    item = {
        **MINIMAL_ITEM,
        "question": "Mennyi a díj? És a határidő?",
        "reference_answer": "Öt.",
    }
    path = tmp_path / "questions.jsonl"
    path.write_text(json.dumps(item, ensure_ascii=False) + "\n", encoding="utf-8")

    (loaded,) = load_dataset(path)

    assert (loaded.question, loaded.reference_answer) == (item["question"], "Öt.")


@pytest.mark.parametrize("content", ["", "\n\n", "  \n\t\n"], ids=["empty", "newlines", "spaces"])
def test_load_dataset_of_a_file_without_items_is_empty(tmp_path: Path, content: str) -> None:
    path = tmp_path / "questions.jsonl"
    path.write_text(content, encoding="utf-8")

    assert load_dataset(path) == []


def test_load_dataset_reports_invalid_json_with_its_line_and_column(tmp_path: Path) -> None:
    error = load_error(tmp_path, '{"id": "q02", "question": }')

    assert isinstance(error.__cause__, json.JSONDecodeError)
    assert error.reason.startswith("invalid JSON: ")
    assert error.reason.endswith(f"(column {error.__cause__.colno})")


def test_load_dataset_rejects_a_repeated_key(tmp_path: Path) -> None:
    error = load_error(
        tmp_path, '{"id": "q02", "id": "q03", "question": "Hello?", "reference_answer": "Hi."}'
    )

    assert error.reason == "duplicate key 'id'"


INVALID_ITEMS = [
    pytest.param('["q02", "Hello?"]', (), "model_type", id="array"),
    pytest.param('"Hello?"', (), "model_type", id="string"),
    pytest.param(
        {"id": "q02", "question": "Hello?"}, ("reference_answer",), "missing", id="missing-field"
    ),
    pytest.param(
        {**MINIMAL_ITEM, "expected_document": ["a.md"]},
        ("expected_document",),
        "extra_forbidden",
        id="unknown-key",
    ),
    pytest.param(
        {**MINIMAL_ITEM, "expected_intent": "lookup"},
        ("expected_intent",),
        "literal_error",
        id="unknown-intent",
    ),
    pytest.param(
        {**MINIMAL_ITEM, "question": "   "}, ("question",), "string_too_short", id="blank-question"
    ),
    pytest.param({**MINIMAL_ITEM, "id": "q 02"}, ("id",), "string_pattern_mismatch", id="bad-id"),
    pytest.param({**MINIMAL_ITEM, "tags": "single-hop"}, ("tags",), "list_type", id="tag-text"),
    pytest.param(
        {**MINIMAL_ITEM, "expected_documents": ["a.md", ""]},
        ("expected_documents", 1),
        "string_too_short",
        id="empty-document",
    ),
    pytest.param(
        {**MINIMAL_ITEM, "expected_documents": ["a.md", "b.md", "a.md"]},
        ("expected_documents",),
        "value_error",
        id="repeated-document",
    ),
]


@pytest.mark.parametrize(("line", "location", "error_type"), INVALID_ITEMS)
def test_load_dataset_reports_an_invalid_item_with_its_line_and_field(
    tmp_path: Path, line: str | dict[str, Any], location: tuple[str | int, ...], error_type: str
) -> None:
    error = load_error(tmp_path, line)

    cause = error.__cause__
    assert isinstance(cause, ValidationError)
    assert [(issue["loc"], issue["type"]) for issue in cause.errors()] == [(location, error_type)]
    field = ".".join(map(str, location))
    assert error.reason.startswith(f"invalid item: {field}: " if field else "invalid item: ")


def test_eval_item_names_the_repeated_entries() -> None:
    with pytest.raises(ValidationError, match=r"contains duplicates: 'a\.md', 'b\.md'"):
        EvalItem.model_validate(
            {**MINIMAL_ITEM, "expected_documents": ["b.md", "a.md", "b.md", "a.md"]}
        )
    with pytest.raises(ValidationError, match="contains duplicates: 'x'"):
        EvalItem.model_validate({**MINIMAL_ITEM, "tags": ["x", "y", "x"]})


def test_load_dataset_rejects_a_repeated_id_and_names_its_first_line(tmp_path: Path) -> None:
    again = {**MINIMAL_ITEM, "question": "Hello again?"}
    path = write_lines(tmp_path / "questions.jsonl", FULL_ITEM, MINIMAL_ITEM, "", again)

    with pytest.raises(DatasetError, match=r":4: duplicate id 'q02', already used on line 2$"):
        load_dataset(path)


def test_load_dataset_reports_text_that_is_not_utf8_with_its_line(tmp_path: Path) -> None:
    legacy = json.dumps({**MINIMAL_ITEM, "question": "Mennyi a díj?"}, ensure_ascii=False)
    path = tmp_path / "questions.jsonl"
    path.write_bytes(json.dumps(FULL_ITEM).encode() + b"\n" + legacy.encode("cp1250") + b"\n")

    with pytest.raises(DatasetError) as caught:
        load_dataset(path)

    assert caught.value.line == 2
    assert caught.value.reason.startswith("not valid UTF-8")
    assert isinstance(caught.value.__cause__, UnicodeDecodeError)


def test_load_dataset_of_a_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "missing.jsonl")


def test_dataset_error_is_a_value_error_that_survives_pickling() -> None:
    error = DatasetError(Path("questions.jsonl"), 7, "invalid JSON")

    restored = pickle.loads(pickle.dumps(error))

    assert isinstance(restored, ValueError)
    assert (restored.path, restored.line) == (Path("questions.jsonl"), 7)
    assert restored.reason == "invalid JSON"
    assert str(restored) == str(error) == f"{Path('questions.jsonl')}:7: invalid JSON"


def test_default_paths_follow_the_planned_layout() -> None:
    assert DEFAULT_DATASET_PATH.as_posix() == "data/eval/questions.jsonl"
    assert RESULTS_DIR.as_posix() == "data/eval/results"
    assert (REPO_ROOT / RESULTS_DIR / ".gitkeep").is_file()


def test_eval_item_is_immutable_and_round_trips_through_json() -> None:
    item = EvalItem.model_validate(FULL_ITEM)

    assert EvalItem.model_validate_json(item.model_dump_json()) == item
    with pytest.raises(ValidationError):
        item.id = "q99"  # type: ignore[misc]


def test_readme_example_is_a_valid_dataset_that_shows_every_field(tmp_path: Path) -> None:
    readme = (REPO_ROOT / "data" / "eval" / "README.md").read_text(encoding="utf-8")
    examples = re.findall(r"```jsonl\n(.*?)```", readme, flags=re.DOTALL)
    assert examples, "data/eval/README.md has no ```jsonl example"
    path = tmp_path / "example.jsonl"
    path.write_text("".join(examples), encoding="utf-8")

    items = load_dataset(path)

    assert items
    shown = {key for line in "".join(examples).splitlines() if line for key in json.loads(line)}
    assert shown == set(EvalItem.model_fields)


# --- deterministic metrics -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("retrieved", "expected", "k", "hit"),
    [
        pytest.param([["a.md", "b.md", "c.md"]], ["c.md"], 3, True, id="hit-at-rank-k"),
        pytest.param([["a.md", "b.md", "c.md"]], ["c.md"], 2, False, id="hit-below-k"),
        pytest.param([["a.md"]], ["z.md", "a.md"], 4, True, id="fewer-chunks-than-k"),
        pytest.param([["a.md", "a.md", "b.md"]], ["b.md"], 2, False, id="ranks-count-chunks"),
        pytest.param([["docs/A.md"]], ["docs/a.md"], 1, False, id="exact-comparison"),
        pytest.param([["x.md", "y.md"], ["b.md"]], ["b.md"], 1, True, id="ranked-per-subtask"),
        pytest.param([["x.md"], ["y.md"]], ["b.md"], 4, False, id="no-subtask-hits"),
        pytest.param([[]], ["a.md"], 4, False, id="nothing-retrieved"),
        pytest.param([], ["a.md"], 4, False, id="no-retrieve-subtask"),
        pytest.param([["a.md"]], [], 4, None, id="not-applicable"),
    ],
)
def test_hit_at_k(
    retrieved: list[list[str]], expected: list[str], k: int, hit: bool | None
) -> None:
    assert hit_at_k(retrieved, expected, k) is hit


def test_hit_at_k_rejects_a_cutoff_below_1_and_input_that_is_not_nested() -> None:
    with pytest.raises(ValueError, match="k must be at least 1, got 0"):
        hit_at_k([["a.md"]], ["a.md"], 0)
    with pytest.raises(TypeError, match="retrieved must be a collection"):
        hit_at_k("a.md", ["a.md"], 1)  # type: ignore[arg-type]
    with pytest.raises(
        TypeError, match=r"one list of document identifiers .* per retrieve sub-task"
    ):
        hit_at_k(["a.md", "b.md"], ["a.md"], 1)
    with pytest.raises(
        TypeError, match=r"one list of document identifiers .* per retrieve sub-task"
    ):
        hit_at_k([[{"source": "a.md"}]], ["a.md"], 1)  # type: ignore[list-item]
    with pytest.raises(TypeError, match="expected must be a collection"):
        hit_at_k([["a.md"]], "a.md", 1)


def test_hit_rate_at_k_averages_the_items_with_expected_documents() -> None:
    retrieved = [[["a.md", "b.md"]], [["c.md"]], [["d.md"]], [["x.md"], ["e.md"]]]
    expected = [["b.md"], ["z.md"], [], ["e.md"]]

    assert hit_rate_at_k(retrieved, expected, k=2) == pytest.approx(2 / 3)
    assert hit_rate_at_k(retrieved, expected, k=1) == pytest.approx(1 / 3)
    assert hit_rate_at_k([[["a.md"]]], [[]], k=3) is None
    assert hit_rate_at_k([], [], k=3) is None


def test_hit_rate_at_k_rejects_unaligned_inputs_and_a_cutoff_below_1() -> None:
    with pytest.raises(ValueError, match="one entry per item, got 2 and 1"):
        hit_rate_at_k([[["a.md"]], [["b.md"]]], [["a.md"]], k=1)
    with pytest.raises(ValueError, match="k must be at least 1"):
        hit_rate_at_k([], [], k=0)


def test_intent_matches_skips_items_without_an_expected_intent() -> None:
    assert intent_matches("single", "single") is True
    assert intent_matches("complex", "single") is False
    assert intent_matches("tool", None) is False
    assert intent_matches(None, "direct") is None


def test_routing_accuracy_averages_the_items_with_an_expected_intent() -> None:
    expected = ["single", "complex", None, "tool", "direct"]
    predicted = ["single", "single", "direct", None, "direct"]

    assert routing_accuracy(expected, predicted) == pytest.approx(2 / 4)
    assert routing_accuracy([None, None], ["direct", "tool"]) is None
    assert routing_accuracy([], []) is None
    with pytest.raises(ValueError, match="one entry per item, got 1 and 0"):
        routing_accuracy(["single"], [])


def test_mean_score_averages_applicable_scores_and_counts_booleans() -> None:
    assert mean_score([1.0, None, 0.5, True, False]) == pytest.approx(0.625)
    assert mean_score(iter([0.25])) == 0.25
    assert mean_score([]) is None
    assert mean_score([None, None]) is None


@pytest.mark.parametrize("score", [-0.1, 1.5, float("nan"), float("inf")])
def test_judge_score_lies_between_0_and_1(score: float) -> None:
    with pytest.raises(ValidationError):
        JudgeScore(score=score)


# --- LLM-judged metrics (Phase 7) ------------------------------------------------------------

JUDGED_METRICS: list[tuple[Callable[..., JudgeScore], tuple[Any, ...]]] = [
    (answer_correctness, ("Question?", "Answer.", "Reference answer.")),
    (faithfulness, ("Answer.", ["Context."])),
]


@pytest.mark.parametrize(("metric", "args"), JUDGED_METRICS, ids=["correctness", "faithfulness"])
def test_llm_judged_metrics_are_planned_for_phase_7(
    metric: Callable[..., JudgeScore], args: tuple[Any, ...]
) -> None:
    from agentic_rag.llm import ScriptedChatModel

    judge = ScriptedChatModel()

    with pytest.raises(PlannedFeatureError) as caught:
        metric(*args, judge=judge)

    assert str(caught.value) == str(planned(f"{metric.__module__}.{metric.__qualname__}", 7))
    assert judge.prompts == ()


# --- item results ----------------------------------------------------------------------------


def result(
    item_id: str,
    *,
    expected_intent: Intent | None = None,
    expected_documents: Sequence[str] = (),
    **fields: Any,
) -> EvalItemResult:
    """Build the result of a question with the given id and expectations."""
    item = EvalItem(
        id=item_id,
        question=f"Question {item_id}?",
        reference_answer="Answer.",
        expected_intent=expected_intent,
        expected_documents=list(expected_documents),
    )
    return EvalItemResult(item=item, **fields)


INCONSISTENT_RESULTS = [
    pytest.param(
        {},
        {"predicted_intent": "single", "intent_correct": True},
        "contradicts the item",
        id="routing-check-without-expected-intent",
    ),
    pytest.param(
        {"expected_intent": "single"},
        {"predicted_intent": "single", "intent_correct": False},
        "contradicts the item",
        id="equal-intents-marked-wrong",
    ),
    pytest.param(
        {"expected_intent": "single"},
        {"predicted_intent": "direct", "intent_correct": True},
        "contradicts the item",
        id="different-intents-marked-right",
    ),
    pytest.param(
        {"expected_intent": "tool"},
        {"intent_correct": True},
        "contradicts the item",
        id="missing-intent-marked-right",
    ),
    pytest.param(
        {},
        {"retrieved_documents": [["a.md"]], "hit_at_k": False},
        "hit_at_k must be None when the item has no expected_documents",
        id="retrieval-check-without-expected-documents",
    ),
    pytest.param(
        {"expected_documents": ["a.md"]},
        {"retrieved_documents": [["b.md"], ["c.md"]], "hit_at_k": True},
        "no expected document was retrieved",
        id="hit-without-an-expected-document",
    ),
    pytest.param(
        {"expected_documents": ["a.md"]},
        {"hit_at_k": True},
        "no expected document was retrieved",
        id="hit-without-retrieval",
    ),
]


@pytest.mark.parametrize(("expectations", "fields", "message"), INCONSISTENT_RESULTS)
def test_eval_item_result_rejects_verdicts_that_contradict_its_item(
    expectations: dict[str, Any], fields: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        result("q01", **expectations, **fields)


CONSISTENT_RESULTS = [
    pytest.param(
        {"expected_intent": "single", "expected_documents": ["a.md"]},
        {"predicted_intent": "single", "retrieved_documents": [["a.md"]]},
        id="metrics-not-computed-for-the-target",
    ),
    pytest.param(
        {"expected_intent": "tool"},
        {"intent_correct": False},
        id="missing-intent-marked-wrong",
    ),
    pytest.param(
        {"expected_documents": ["a.md"]},
        {"retrieved_documents": [["b.md", "c.md", "a.md"]], "hit_at_k": False},
        id="expected-document-below-k",
    ),
    pytest.param(
        {"expected_documents": ["a.md"]},
        {"retrieved_documents": [["b.md"], ["a.md"]], "hit_at_k": True},
        id="hit-in-a-later-subtask",
    ),
    pytest.param(
        {"expected_documents": ["a.md"]},
        {"hit_at_k": False, "error": "TimeoutError: no answer"},
        id="failed-run-is-a-miss",
    ),
]


@pytest.mark.parametrize(("expectations", "fields"), CONSISTENT_RESULTS)
def test_eval_item_result_accepts_verdicts_that_agree_with_its_item(
    expectations: dict[str, Any], fields: dict[str, Any]
) -> None:
    outcome = result("q01", **expectations, **fields)

    assert outcome.model_dump(include=set(fields)) == fields


def test_eval_item_result_rejects_non_finite_latency() -> None:
    with pytest.raises(ValidationError):
        result("q01", latency_ms=float("inf"))


# --- runner ----------------------------------------------------------------------------------


def test_node_targets_are_the_main_graph_nodes_an_item_can_drive() -> None:
    from agentic_rag.agent.graph import NODE_NAMES

    assert NODE_TARGETS == ("analyze_request", "run_rag_subtask")
    assert sorted(NODE_TARGETS + NODES_WITHOUT_ITEM_INPUT) == sorted(NODE_NAMES)


@pytest.mark.parametrize(
    "arguments",
    [{}, *({"target": "node", "node": node} for node in NODE_TARGETS)],
    ids=["graph", *NODE_TARGETS],
)
def test_run_evaluation_is_planned_for_phase_7(
    settings: Settings, tmp_path: Path, arguments: dict[str, Any]
) -> None:
    with pytest.raises(PlannedFeatureError) as caught:
        run_evaluation(
            settings, dataset_path=tmp_path / "questions.jsonl", output_dir=tmp_path, **arguments
        )

    assert str(caught.value) == str(planned("agentic_rag.evaluation.runner.run_evaluation", 7))


@pytest.mark.parametrize("node", [*NODES_WITHOUT_ITEM_INPUT, "no_such_node", ""])
def test_run_evaluation_rejects_a_node_that_an_item_cannot_drive(
    settings: Settings, node: str
) -> None:
    with pytest.raises(InvalidArgumentError) as caught:
        run_evaluation(settings, target="node", node=node)

    message = str(caught.value)
    assert message.startswith(f"node {node!r} cannot be evaluated on its own")
    assert all(target in message for target in NODE_TARGETS)


@pytest.mark.parametrize(
    ("target", "node"),
    [("node", None), ("graph", "analyze_request"), ("graphs", None)],
    ids=["node-without-name", "graph-with-name", "unknown-target"],
)
def test_run_evaluation_rejects_a_node_that_does_not_fit_the_target(
    settings: Settings, target: Any, node: str | None
) -> None:
    with pytest.raises(InvalidArgumentError):
        run_evaluation(settings, target=target, node=node)


def test_run_evaluation_signature_matches_the_cli_call() -> None:
    signature = inspect.signature(run_evaluation)
    settings_parameter, *options = signature.parameters.values()

    assert settings_parameter.name == "settings"
    assert settings_parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert settings_parameter.default is inspect.Parameter.empty
    assert [(option.name, option.kind, option.default) for option in options] == [
        ("target", inspect.Parameter.KEYWORD_ONLY, "graph"),
        ("node", inspect.Parameter.KEYWORD_ONLY, None),
        ("dataset_path", inspect.Parameter.KEYWORD_ONLY, None),
        ("output_dir", inspect.Parameter.KEYWORD_ONLY, None),
    ]
    assert signature.return_annotation is EvalReport


# --- report ----------------------------------------------------------------------------------


def sample_report(settings: Settings) -> EvalReport:
    """A graph evaluation of three questions: a full result, a misrouted one and a failure.

    Every verdict agrees with its item, as ``EvalItemResult`` requires.
    """
    items = [
        result(
            "q01",
            expected_intent="single",
            expected_documents=["guides/a.md"],
            answer="Answer one.",
            predicted_intent="single",
            retrieved_documents=[["guides/a.md", "guides/b.md"]],
            intent_correct=True,
            hit_at_k=True,
            answer_correctness=JudgeScore(score=1.0, rationale="Matches the reference."),
            faithfulness=JudgeScore(score=0.75),
            latency_ms=120.0,
        ),
        # Answered directly, so nothing was retrieved: a routing error and a retrieval miss.
        result(
            "q02",
            expected_intent="single",
            expected_documents=["guides/c.md"],
            answer="Answer two.",
            predicted_intent="direct",
            intent_correct=False,
            hit_at_k=False,
            answer_correctness=JudgeScore(score=0.5),
            latency_ms=80.0,
        ),
        # Routing is not checked for this question; the failed run retrieved nothing.
        result(
            "q03",
            expected_documents=["guides/d.md"],
            hit_at_k=False,
            error="TimeoutError: no answer",
        ),
    ]
    return EvalReport(
        created_at=datetime(2026, 10, 1, 12, 0, tzinfo=UTC),
        target="graph",
        dataset=Path("data/eval/questions.jsonl"),
        settings=settings,
        items=items,
    )


def test_eval_report_computes_its_scores_from_the_applicable_items(settings: Settings) -> None:
    report = sample_report(settings)

    scores = report.scores
    assert scores.routing_accuracy == MetricSummary(mean=0.5, count=2)
    assert scores.hit_rate_at_k.mean == pytest.approx(1 / 3)
    assert scores.hit_rate_at_k.count == 3
    assert scores.answer_correctness == MetricSummary(mean=0.75, count=2)
    assert scores.faithfulness == MetricSummary(mean=0.75, count=1)
    assert report.error_count == 1


def test_eval_report_without_items_has_no_scores() -> None:
    report = EvalReport(target="graph", dataset="questions.jsonl", settings={}, items=[])

    assert report.scores.routing_accuracy == MetricSummary(mean=None, count=0)
    assert report.scores.faithfulness == MetricSummary(mean=None, count=0)
    assert report.error_count == 0


def test_eval_report_round_trips_through_json(settings: Settings) -> None:
    report = sample_report(settings)

    data = json.loads(report.model_dump_json())

    assert data["created_at"] == "2026-10-01T12:00:00Z"
    assert data["dataset"] == "data/eval/questions.jsonl"
    assert data["settings"] == settings.as_env()
    assert data["settings"]["LLM_PROVIDER"] == "fake"
    assert data["scores"]["routing_accuracy"] == {"mean": 0.5, "count": 2}
    assert data["error_count"] == 1
    assert data["items"][0]["item"]["expected_documents"] == ["guides/a.md"]
    assert data["items"][0]["retrieved_documents"] == [["guides/a.md", "guides/b.md"]]
    assert EvalReport.model_validate_json(report.model_dump_json()) == report


@pytest.mark.parametrize(
    ("target", "node"),
    [("node", None), ("node", ""), ("node", "verify_answer"), ("graph", "analyze_request")],
    ids=[
        "node-without-name",
        "node-with-empty-name",
        "node-an-item-cannot-drive",
        "graph-with-name",
    ],
)
def test_eval_report_names_an_evaluable_node_exactly_for_node_targets(
    target: Any, node: str | None
) -> None:
    with pytest.raises(ValidationError):
        EvalReport(target=target, node=node, dataset="questions.jsonl", settings={}, items=[])


@pytest.mark.parametrize("node", NODE_TARGETS)
def test_eval_report_of_a_node_target_keeps_the_node_name(node: str) -> None:
    report = EvalReport(target="node", node=node, dataset="questions.jsonl", settings={}, items=[])

    assert (report.target, report.node) == ("node", node)


def test_eval_report_is_a_run_report_with_a_settings_snapshot_and_a_utc_timestamp(
    settings: Settings,
) -> None:
    report = EvalReport(target="graph", dataset="questions.jsonl", settings=settings, items=[])

    assert isinstance(report, RunReport)
    assert report.settings == settings.as_env()
    assert report.created_at.utcoffset() == timedelta(0)


def test_eval_report_rejects_a_naive_timestamp() -> None:
    with pytest.raises(ValidationError) as caught:
        EvalReport(
            created_at=datetime(2026, 10, 1, 12, 0),  # naive on purpose
            target="graph",
            dataset="questions.jsonl",
            settings={},
            items=[],
        )

    assert [(issue["loc"], issue["type"]) for issue in caught.value.errors()] == [
        (("created_at",), "timezone_aware")
    ]


@pytest.mark.parametrize(
    "fields",
    [{"mean": 0.5, "count": 0}, {"mean": None, "count": 2}, {"mean": 1.5, "count": 2}],
    ids=["mean-without-items", "items-without-mean", "mean-above-1"],
)
def test_metric_summary_keeps_mean_and_count_consistent(fields: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        MetricSummary(**fields)
