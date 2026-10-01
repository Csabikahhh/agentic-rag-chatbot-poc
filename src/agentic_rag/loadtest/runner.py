"""Load-test runner: latency statistics, the report model and ``run_load_test``.

Real already, and pure: :class:`LatencyStats` with :func:`summarize_latencies` and
:func:`percentile`, the per-node aggregation :func:`summarize_node_latencies`, and
:class:`LoadTestReport`, the format of the committed result files. ``run_load_test`` itself is
planned for Phase 8. The records reject NaN and infinite numbers, so every report serializes
to valid JSON and loads back unchanged.

Percentile method: linear interpolation between the closest ranks. For ``n`` sorted samples
``x[0] <= ... <= x[n - 1]`` and a percentile ``q`` from 0 to 100, the position
``h = (n - 1) * q / 100`` lies between the ranks ``floor(h)`` and ``ceil(h)``, and the result
is ``x[floor(h)] + (h - floor(h)) * (x[ceil(h)] - x[floor(h)])``. This is method 7 of Hyndman
and Fan (1996), the default of ``numpy.percentile``, Excel's ``PERCENTILE.INC`` and
``statistics.quantiles(..., method="inclusive")``, so every reported number can be checked
with any of them. p50 is the ordinary median, p0 the minimum and p100 the maximum. A high
percentile of a small sample interpolates between the slowest requests: with 100 samples,
p99 lies between the two slowest.

What the statistics cover (the contract for Phase 8):

- The latency of a request is the wall-clock time of one ``graph.invoke`` call of the
  compiled main graph, in milliseconds.
- Warm-up requests are excluded from every measured statistic and summarized on their own in
  ``LoadTestReport.warmup_latency``: the first Ollama request also loads the model.
- The end-to-end statistics cover the successful measured requests. A failed request counts
  only towards ``error_count`` and the error rate, because its latency describes the failure
  rather than the system.
- The per-node statistics come from the traces of the successful measured requests, one
  sample per node execution. A node that runs several times in one request (``Send``
  fan-out, re-planning) contributes several samples, so its ``count`` divided by the number
  of requests is its executions per request. ``run_rag_subtask`` is timed inclusively: its
  duration contains the RAG subgraph nodes it ran, and those appear under their own names as
  well, so never add a parent node and its subgraph nodes together when computing time
  shares.
"""

import itertools
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from agentic_rag.config import Settings
from agentic_rag.errors import planned
from agentic_rag.reports import RunReport
from agentic_rag.tracing import TraceEvent

__all__ = [
    "LatencyStats",
    "LoadTestReport",
    "percentile",
    "run_load_test",
    "summarize_latencies",
    "summarize_node_latencies",
]


class LatencyStats(BaseModel):
    """Summary statistics of latency samples, in milliseconds.

    With no samples (``count == 0``) every statistic is ``None``, so an empty summary is never
    mistaken for a fast one. Otherwise every statistic is set and they are ordered:
    ``min_ms <= p50_ms <= p95_ms <= p99_ms <= max_ms``, with ``mean_ms`` between ``min_ms`` and
    ``max_ms``. Build instances with :func:`summarize_latencies`; they are immutable values.
    """

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    count: int = Field(ge=0, description="Number of samples.")
    mean_ms: float | None = Field(default=None, ge=0, description="Arithmetic mean.")
    min_ms: float | None = Field(default=None, ge=0, description="Fastest sample.")
    p50_ms: float | None = Field(default=None, ge=0, description="50th percentile (median).")
    p95_ms: float | None = Field(default=None, ge=0, description="95th percentile.")
    p99_ms: float | None = Field(default=None, ge=0, description="99th percentile.")
    max_ms: float | None = Field(default=None, ge=0, description="Slowest sample.")

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        """Reject statistics that no set of samples can produce."""
        ordered = (self.min_ms, self.p50_ms, self.p95_ms, self.p99_ms, self.max_ms)
        present = [value for value in ordered if value is not None]
        if self.count == 0:
            if present or self.mean_ms is not None:
                raise ValueError("every statistic must be None when count is 0")
            return self
        if len(present) < len(ordered) or self.mean_ms is None:
            raise ValueError("every statistic must be set when count is positive")
        if any(low > high for low, high in itertools.pairwise(present)):
            raise ValueError("statistics must satisfy min <= p50 <= p95 <= p99 <= max")
        if not present[0] <= self.mean_ms <= present[-1]:
            raise ValueError("mean_ms must lie between min_ms and max_ms")
        return self


class LoadTestReport(RunReport):
    """Result of one load-test run: the format of the load-test files in ``data/eval/results/``.

    ``created_at`` and the ``settings`` snapshot come from :class:`~agentic_rag.reports.RunReport`;
    ``settings["LLM_PROVIDER"]`` tells a fake-mode baseline from an Ollama run. Every measured
    statistic excludes the warm-up requests, which ``warmup_latency`` summarizes separately;
    the module docstring defines what each statistic covers. ``error_rate`` and
    ``throughput_rps`` are computed from the other fields. They appear in the JSON output, and
    loading a report back recomputes them instead of reading them. Reports are immutable
    values.
    """

    request_count: int = Field(ge=1, description="Measured requests, warm-up excluded.")
    concurrency: int = Field(ge=1, description="Maximum number of requests in flight at once.")
    warmup_count: int = Field(ge=0, description="Warm-up requests sent before the measured ones.")
    error_count: int = Field(ge=0, description="Measured requests that failed.")
    duration_s: float = Field(
        gt=0,
        description="Wall-clock time of the measured phase in seconds, from the start of the "
        "first measured request to the end of the last one.",
    )
    latency: LatencyStats = Field(
        description="End-to-end latency of the successful measured requests."
    )
    node_latency: dict[str, LatencyStats] = Field(
        description="Latency per node execution in the successful measured requests, keyed by "
        "node name in order of first appearance (summarize_node_latencies)."
    )
    warmup_latency: LatencyStats = Field(
        description="End-to-end latency of the successful warm-up requests, reported apart."
    )

    @model_validator(mode="after")
    def _check_counts(self) -> Self:
        """Keep the request counts and the sample counts of the statistics consistent."""
        if self.error_count > self.request_count:
            raise ValueError("error_count cannot exceed request_count")
        if self.latency.count != self.request_count - self.error_count:
            raise ValueError("latency must summarize exactly the successful measured requests")
        if self.warmup_latency.count > self.warmup_count:
            raise ValueError("warmup_latency cannot have more samples than warmup_count")
        return self

    @computed_field
    @property
    def error_rate(self) -> float:
        """Share of the measured requests that failed, from 0 to 1."""
        return self.error_count / self.request_count

    @computed_field
    @property
    def throughput_rps(self) -> float:
        """Measured requests completed per second of the measured phase, failed ones included."""
        return self.request_count / self.duration_s


def percentile(values: Sequence[float], q: float) -> float:
    """Return the ``q``-th percentile of ``values``, interpolating linearly between ranks.

    The module docstring defines the method; it matches ``numpy.percentile`` with its default
    settings.

    Args:
        values: The samples in any order; at least one, every one a finite number.
        q: The percentile, from 0 to 100 inclusive.

    Returns:
        The percentile, in the unit of ``values``.

    Raises:
        ValueError: If ``q`` is outside [0, 100], ``values`` is empty, or a value is NaN or
            infinite.
    """
    _check_percentile(q)
    samples = sorted(_samples(values, non_negative=False))
    if not samples:
        raise ValueError("percentile() needs at least one value")
    return _interpolate(samples, q)


def summarize_latencies(values_ms: Sequence[float]) -> LatencyStats:
    """Summarize latency samples: count, mean, min, p50, p95, p99 and max.

    Args:
        values_ms: Latencies in milliseconds in any order; every one finite and not negative.

    Returns:
        The statistics, with percentiles as :func:`percentile` computes them. An empty input
        gives ``LatencyStats(count=0)``, whose statistics are all ``None``.

    Raises:
        ValueError: If a value is negative, NaN or infinite.
    """
    samples = sorted(_samples(values_ms, non_negative=True))
    if not samples:
        return LatencyStats(count=0)
    fastest, slowest = samples[0], samples[-1]
    mean = math.fsum(samples) / len(samples)
    return LatencyStats(
        count=len(samples),
        # Rounding can put the mean of equal samples one ulp outside them; keep it inside.
        mean_ms=min(max(mean, fastest), slowest),
        min_ms=fastest,
        p50_ms=_interpolate(samples, 50.0),
        p95_ms=_interpolate(samples, 95.0),
        p99_ms=_interpolate(samples, 99.0),
        max_ms=slowest,
    )


def summarize_node_latencies(traces: Iterable[Sequence[TraceEvent]]) -> dict[str, LatencyStats]:
    """Summarize the latency of every node from the traces of several requests.

    Every event is one sample of its node, so a node that ran several times in one request
    contributes several samples (see the module docstring).

    Args:
        traces: One trace per request, e.g. ``graph.invoke(...)["trace"]``, each holding
            :class:`~agentic_rag.tracing.TraceEvent` objects in execution order. Pass the
            traces of the successful measured requests only, without the warm-up.

    Returns:
        The statistics of ``TraceEvent.duration_ms`` per node, keyed by node name in order of
        first appearance; empty when there are no events.

    Raises:
        TypeError: If ``traces`` is a single flat trace instead of one trace per request, or
            a trace holds something other than ``TraceEvent`` objects.
    """
    durations: dict[str, list[float]] = {}
    for trace in traces:
        if isinstance(trace, TraceEvent):
            raise TypeError("traces must hold one sequence of TraceEvent per request")
        for event in trace:
            if not isinstance(event, TraceEvent):
                raise TypeError(f"a trace must hold TraceEvent objects, got {type(event).__name__}")
            durations.setdefault(event.node, []).append(event.duration_ms)
    return {node: summarize_latencies(samples) for node, samples in durations.items()}


def run_load_test(
    settings: Settings,
    *,
    requests: int = 100,
    concurrency: int = 4,
    warmup: int = 3,
    output_dir: Path | None = None,
) -> LoadTestReport:
    """Send queries to the compiled main graph under load and measure them (Phase 8).

    Planned behaviour, with the details settled in Phase 8:

    1. Build the main graph once with ``agentic_rag.agent.graph.build_agent_graph(settings)``
       and take the questions of the evaluation set in turn.
    2. Send the ``warmup`` requests and wait for them to finish. They are left out of every
       measured number (``duration_s``, ``latency``, ``node_latency`` and the error count)
       and summarized in ``warmup_latency`` instead, because the first Ollama request also
       loads the model and would distort the percentiles.
    3. Send the ``requests`` measured requests from a
       ``concurrent.futures.ThreadPoolExecutor(max_workers=concurrency)``, every task calling
       ``graph.invoke``, so at most ``concurrency`` requests are in flight. Time every request
       on the trace clock (``agentic_rag.tracing.epoch_now``) and keep its trace.
    4. Build the report with :func:`summarize_latencies` and :func:`summarize_node_latencies`
       and write it as JSON to ``output_dir``.

    Why threads and ``invoke`` rather than ``asyncio`` and ``ainvoke``: the nodes are
    synchronous functions, and under ``ainvoke`` LangGraph runs every one of them in a worker
    thread anyway, a hop per node that no node's ``duration_ms`` contains. With ``invoke`` a request
    takes the same synchronous path as in the UI and the evaluation. An Ollama call waits on
    the network with the GIL released, so the threads overlap the way concurrent clients do;
    in fake mode the work is pure Python and the threads share the GIL, so the fake baseline
    shows the framework's own cost rather than a parallel speed-up. One run with
    ``LLM_PROVIDER=fake`` and one with ``ollama`` separate the LLM's share of the latency from
    the rest of the system.

    Args:
        settings: The effective settings; the report records them.
        requests: Number of measured requests, at least 1; the assignment asks for 50-200.
        concurrency: Maximum number of requests in flight at the same time, at least 1; the
            number of worker threads.
        warmup: Number of warm-up requests, sent first and reported separately; 0 or more.
        output_dir: Directory for the JSON report; ``None`` means
            ``agentic_rag.reports.RESULTS_DIR``.

    Returns:
        The report, as written to ``output_dir``.

    Raises:
        PlannedFeatureError: Always, until Phase 8.
    """
    raise planned(f"{__name__}.run_load_test", 8)


def _samples(values: Iterable[float], *, non_negative: bool) -> list[float]:
    """Convert samples to floats, rejecting NaN, infinities and, optionally, negatives."""
    samples = [float(value) for value in values]
    for value in samples:
        if not math.isfinite(value):
            raise ValueError(f"samples must be finite numbers, got {value}")
        if non_negative and value < 0:
            raise ValueError(f"latencies cannot be negative, got {value}")
    return samples


def _check_percentile(q: float) -> None:
    """Reject a percentile outside [0, 100]; the comparison also rejects NaN."""
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"q must be between 0 and 100, got {q}")


def _interpolate(ordered: Sequence[float], q: float) -> float:
    """Return the ``q``-th percentile of non-empty, ascending samples (module docstring)."""
    _check_percentile(q)
    position = (len(ordered) - 1) * q / 100.0
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    low, high = ordered[lower], ordered[upper]
    value = low + (position - lower) * (high - low)
    # Rounding can push the result a hair outside [low, high]; clamping keeps every
    # percentile between its neighbouring samples, so p50 <= p95 <= p99 always holds.
    return min(max(value, low), high)
