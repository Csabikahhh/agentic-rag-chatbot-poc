"""Tests for agentic_rag.loadtest.runner: percentiles, latency summaries and the report model.

The percentile tests compare with hand-computed values of the documented method (linear
interpolation between the closest ranks), with statistics.quantiles(method="inclusive") and,
when it is installed, with numpy.percentile. Everything is pure and offline; the Phase 8 stub
is checked for its planned-phase error and for the call signature that the CLI uses.
"""

import inspect
import json
import math
import random
import statistics
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from agentic_rag.config import Settings
from agentic_rag.errors import PlannedFeatureError, planned
from agentic_rag.loadtest.runner import (
    LatencyStats,
    LoadTestReport,
    percentile,
    run_load_test,
    summarize_latencies,
    summarize_node_latencies,
)
from agentic_rag.reports import RunReport
from agentic_rag.tracing import TraceEvent


def event(node: str, duration_ms: float, started_at: float = 1_000.0) -> TraceEvent:
    """Build a trace event of the given node and duration."""
    return TraceEvent(
        node=node,
        started_at=started_at,
        ended_at=started_at + duration_ms / 1000.0,
        duration_ms=duration_ms,
    )


def lognormal_samples(size: int, seed: int) -> list[float]:
    """Latency-like samples: positive and right-skewed, reproducible for a seed."""
    rng = random.Random(seed)
    return [rng.lognormvariate(5.0, 0.8) for _ in range(size)]


# --- percentile ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("q", "expected"),
    [(0, 10.0), (25, 17.5), (50, 25.0), (95, 38.5), (99, 39.7), (100, 40.0)],
)
def test_percentile_interpolates_between_the_closest_ranks(q: float, expected: float) -> None:
    # Sorted samples 10, 20, 30, 40: position h = 3 * q / 100, so q = 95 gives h = 2.85 and
    # 30 + 0.85 * (40 - 30) = 38.5; q = 50 gives h = 1.5, the ordinary median 25.
    assert percentile([40, 10, 30, 20], q) == pytest.approx(expected)


@pytest.mark.parametrize("q", [0, 50, 99, 100])
def test_percentile_of_a_single_sample_is_that_sample(q: float) -> None:
    assert percentile([7.5], q) == 7.5


@pytest.mark.parametrize("size", [2, 7, 50, 100, 137])
def test_percentile_agrees_with_statistics_quantiles_inclusive(size: int) -> None:
    samples = lognormal_samples(size, seed=size)

    cut_points = statistics.quantiles(samples, n=100, method="inclusive")

    assert [percentile(samples, q) for q in range(1, 100)] == pytest.approx(cut_points)


def test_percentile_agrees_with_numpy_percentile() -> None:
    numpy = pytest.importorskip("numpy")
    samples = lognormal_samples(137, seed=7)

    for q in (0, 5, 50, 90, 95, 99, 99.9, 100):
        assert percentile(samples, q) == pytest.approx(float(numpy.percentile(samples, q)))


@pytest.mark.parametrize(
    ("values", "q", "message"),
    [
        ([], 50, "at least one value"),
        ([1.0], -1, "between 0 and 100"),
        ([1.0], 100.5, "between 0 and 100"),
        ([1.0], math.nan, "between 0 and 100"),
        ([1.0, math.nan], 50, "finite"),
        ([1.0, math.inf], 50, "finite"),
    ],
    ids=["empty", "below-0", "above-100", "nan-q", "nan-sample", "infinite-sample"],
)
def test_percentile_rejects_invalid_input(values: list[float], q: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        percentile(values, q)


# --- summarize_latencies ---------------------------------------------------------------------


def test_summarize_latencies_matches_hand_computed_values() -> None:
    # Sorted 15, 20, 35, 40, 50: mean 160 / 5 = 32 and h = 4 * q / 100, so p50 is h = 2 -> 35,
    # p95 is h = 3.8 -> 40 + 0.8 * 10 = 48 and p99 is h = 3.96 -> 40 + 0.96 * 10 = 49.6.
    stats = summarize_latencies([50, 15, 40, 20, 35])

    assert stats.model_dump() == pytest.approx(
        {
            "count": 5,
            "mean_ms": 32.0,
            "min_ms": 15.0,
            "p50_ms": 35.0,
            "p95_ms": 48.0,
            "p99_ms": 49.6,
            "max_ms": 50.0,
        }
    )


def test_summarize_latencies_of_100_requests() -> None:
    # 1..100 ms and h = 99 * q / 100: p50 is h = 49.5 -> 50.5, p95 is h = 94.05 -> 95.05 and
    # p99 is h = 98.01 -> 99.01, between the two slowest requests.
    stats = summarize_latencies([float(value) for value in range(100, 0, -1)])

    assert (stats.count, stats.min_ms, stats.max_ms) == (100, 1.0, 100.0)
    assert (stats.mean_ms, stats.p50_ms, stats.p95_ms, stats.p99_ms) == pytest.approx(
        (50.5, 50.5, 95.05, 99.01)
    )


def test_summarize_latencies_of_a_single_sample() -> None:
    assert summarize_latencies([12.5]) == LatencyStats(
        count=1, mean_ms=12.5, min_ms=12.5, p50_ms=12.5, p95_ms=12.5, p99_ms=12.5, max_ms=12.5
    )


def test_summarize_latencies_without_samples_is_explicitly_empty() -> None:
    stats = summarize_latencies([])

    assert stats == LatencyStats(count=0)
    assert stats.model_dump() == {
        "count": 0,
        "mean_ms": None,
        "min_ms": None,
        "p50_ms": None,
        "p95_ms": None,
        "p99_ms": None,
        "max_ms": None,
    }


def test_summarize_latencies_keeps_the_mean_within_the_samples() -> None:
    # The float mean of three 0.1 samples is 0.10000000000000002, just above the maximum.
    assert math.fsum([0.1] * 3) / 3 > 0.1

    assert summarize_latencies([0.1] * 3).mean_ms == 0.1


def test_summarize_latencies_always_yields_ordered_statistics() -> None:
    rng = random.Random(42)
    for size in range(1, 80):
        samples = [
            rng.choice([0.1, 0.2, 0.3, rng.random(), rng.expovariate(0.01)]) for _ in range(size)
        ]

        stats = summarize_latencies(samples)

        assert stats.count == size
        assert stats.min_ms == min(samples)
        assert stats.max_ms == max(samples)


@pytest.mark.parametrize(
    "values", [[-1.0], [1.0, math.nan], [math.inf]], ids=["negative", "nan", "infinite"]
)
def test_summarize_latencies_rejects_invalid_samples(values: list[float]) -> None:
    with pytest.raises(ValueError, match=r"negative|finite"):
        summarize_latencies(values)


ORDERED = {
    "mean_ms": 5.0,
    "min_ms": 1.0,
    "p50_ms": 4.0,
    "p95_ms": 8.0,
    "p99_ms": 9.0,
    "max_ms": 9.0,
}


@pytest.mark.parametrize(
    "fields",
    [
        {"count": 0, "mean_ms": 1.0},
        {"count": 2},
        {"count": 2, **ORDERED, "p99_ms": None},
        {"count": 2, **ORDERED, "p95_ms": 3.0},
        {"count": 2, **ORDERED, "mean_ms": 10.0},
        {"count": 2, **ORDERED, "min_ms": -1.0},
        {"count": 2, **ORDERED, "max_ms": math.inf},
        {"count": -1},
    ],
    ids=[
        "values-without-samples",
        "samples-without-values",
        "missing-value",
        "unordered-percentiles",
        "mean-above-max",
        "negative",
        "infinite",
        "negative-count",
    ],
)
def test_latency_stats_rejects_impossible_statistics(fields: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        LatencyStats(**fields)


# --- summarize_node_latencies ----------------------------------------------------------------


def test_summarize_node_latencies_from_synthetic_traces() -> None:
    traces = [
        [
            event("analyze_request", 10),
            event("plan_subtasks", 20),
            event("retrieve", 5),
            event("run_rag_subtask", 30),
            event("run_rag_subtask", 50),
            event("synthesize_answer", 100),
        ],
        (event("analyze_request", 30), event("finalize_response", 4)),
    ]

    stats = summarize_node_latencies(traces)

    assert list(stats) == [
        "analyze_request",
        "plan_subtasks",
        "retrieve",
        "run_rag_subtask",
        "synthesize_answer",
        "finalize_response",
    ]
    assert stats["analyze_request"] == summarize_latencies([10, 30])
    assert (stats["run_rag_subtask"].count, stats["run_rag_subtask"].p50_ms) == (2, 40.0)
    assert stats["finalize_response"] == summarize_latencies([4])


def test_summarize_node_latencies_accepts_any_iterable_of_traces() -> None:
    traces = ([event("retrieve", float(index))] for index in range(1, 4))

    assert summarize_node_latencies(traces) == {"retrieve": summarize_latencies([1.0, 2.0, 3.0])}


def test_summarize_node_latencies_without_events_is_empty() -> None:
    assert summarize_node_latencies([]) == {}
    assert summarize_node_latencies([[], ()]) == {}


def test_summarize_node_latencies_rejects_a_flat_trace_and_foreign_items() -> None:
    with pytest.raises(TypeError, match="one sequence of TraceEvent per request"):
        summarize_node_latencies([event("retrieve", 5)])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="got dict"):
        summarize_node_latencies([[{"node": "retrieve", "duration_ms": 5.0}]])  # type: ignore[list-item]


# --- report ----------------------------------------------------------------------------------


def report_fields(settings: Settings, **overrides: Any) -> dict[str, Any]:
    """Valid report fields: 100 measured requests of which 4 failed, after 3 warm-up requests."""
    fields: dict[str, Any] = {
        "created_at": datetime(2026, 10, 1, 12, 0, tzinfo=UTC),
        "request_count": 100,
        "concurrency": 4,
        "warmup_count": 3,
        "error_count": 4,
        "duration_s": 50.0,
        "latency": summarize_latencies([float(value) for value in range(1, 97)]),
        "node_latency": summarize_node_latencies(
            [[event("analyze_request", 10), event("synthesize_answer", 900)]]
        ),
        "warmup_latency": summarize_latencies([3000.0, 800.0, 750.0]),
        "settings": settings,
    }
    fields.update(overrides)
    return fields


def test_load_test_report_derives_error_rate_and_throughput(settings: Settings) -> None:
    report = LoadTestReport(**report_fields(settings))

    assert report.error_rate == pytest.approx(0.04)
    assert report.throughput_rps == pytest.approx(2.0)
    assert report.settings == settings.as_env()


def test_load_test_report_is_a_run_report_with_a_utc_timestamp(settings: Settings) -> None:
    fields = report_fields(settings)
    del fields["created_at"]

    report = LoadTestReport(**fields)

    assert isinstance(report, RunReport)
    assert report.created_at.utcoffset() == timedelta(0)


def test_load_test_report_round_trips_through_json(settings: Settings) -> None:
    report = LoadTestReport(**report_fields(settings))

    data = json.loads(report.model_dump_json())

    assert (data["error_rate"], data["throughput_rps"]) == (0.04, 2.0)
    assert data["created_at"] == "2026-10-01T12:00:00Z"
    assert list(data["node_latency"]) == ["analyze_request", "synthesize_answer"]
    assert data["warmup_latency"]["max_ms"] == 3000.0
    assert data["settings"]["LLM_PROVIDER"] == "fake"
    assert LoadTestReport.model_validate_json(report.model_dump_json()) == report


@pytest.mark.parametrize(
    "overrides",
    [
        {"error_count": 101},
        {"error_count": 0},
        {"warmup_count": 2},
        {"request_count": 0},
        {"concurrency": 0},
        {"duration_s": 0.0},
        {"duration_s": math.inf},
        {"created_at": datetime(2026, 10, 1, 12, 0)},  # naive on purpose
    ],
    ids=[
        "more-errors-than-requests",
        "latency-count-mismatch",
        "more-warmup-samples-than-requests",
        "no-requests",
        "no-concurrency",
        "zero-duration",
        "infinite-duration",
        "naive-timestamp",
    ],
)
def test_load_test_report_rejects_inconsistent_values(
    settings: Settings, overrides: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError):
        LoadTestReport(**report_fields(settings, **overrides))


# --- run_load_test (Phase 8) -----------------------------------------------------------------


def test_run_load_test_is_planned_for_phase_8(settings: Settings, tmp_path: Path) -> None:
    expected = str(planned("agentic_rag.loadtest.runner.run_load_test", 8))

    with pytest.raises(PlannedFeatureError) as caught:
        run_load_test(settings, requests=50, concurrency=8, warmup=0, output_dir=tmp_path)
    assert str(caught.value) == expected

    with pytest.raises(PlannedFeatureError) as caught:
        run_load_test(settings)
    assert str(caught.value) == expected


def test_run_load_test_signature_matches_the_cli_call() -> None:
    signature = inspect.signature(run_load_test)
    settings_parameter, *options = signature.parameters.values()

    assert settings_parameter.name == "settings"
    assert settings_parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert settings_parameter.default is inspect.Parameter.empty
    assert [(option.name, option.kind, option.default) for option in options] == [
        ("requests", inspect.Parameter.KEYWORD_ONLY, 100),
        ("concurrency", inspect.Parameter.KEYWORD_ONLY, 4),
        ("warmup", inspect.Parameter.KEYWORD_ONLY, 3),
        ("output_dir", inspect.Parameter.KEYWORD_ONLY, None),
    ]
    assert signature.return_annotation is LoadTestReport
