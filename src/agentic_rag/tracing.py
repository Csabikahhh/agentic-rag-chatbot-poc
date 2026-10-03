"""Step-trace primitives shared by the UI, the evaluation and the load test.

Every LangGraph node in the project is wrapped with :func:`traced`. The decorator times the
node and appends one :class:`TraceEvent` to the node's partial state update under the
``"trace"`` key (:data:`TRACE_KEY`), so the event travels through the graph like any other
update:

- ``AgentState`` and ``RagState`` declare ``trace`` with an append reducer, so the final state
  returned by ``graph.invoke`` holds every event of the run. The evaluation and the load test
  read per-node latency from there.
- Every ``graph.stream(..., stream_mode="updates")`` chunk carries the events of the node that
  has just finished. :func:`trace_events_from_chunk` extracts them for the UI step panel.

The contract assumes the node convention of both graphs: a node returns a ``dict`` partial
update or ``None``. :func:`traced` raises ``TypeError`` for any other return value,
``Command`` included; ``Command`` support arrives in the phase that first returns one. That
phase also needs a rule for ``Command(graph=Command.PARENT)`` from a subgraph node: its update
reaches the stream only under the parent node's key, where ``skip_forwarded`` drops it, and
the subgraph's earlier events never reach the final state.

Subgraph events: a node that invokes a subgraph (``run_rag_subtask`` invoking the RAG
subgraph) forwards the subgraph's ``trace`` in its own update, so ``invoke`` results include
the subgraph's per-node timings as well. When streaming with ``subgraphs=True``, the
subgraph's own chunks deliver those events live and the parent node's chunk repeats them;
call ``trace_events_from_chunk(chunk, skip_forwarded=True)`` in that mode, so that the
forwarded events are not counted twice.

Caching: do not give a traced node a LangGraph ``CachePolicy``. A cache hit replays the
node's cached writes, the old event included, so the trace would report the original start
time and duration for a step that took no time. Cache inside the node instead (for example,
memoize the vector search by query), so that every execution records a fresh event.

Timestamps: ``started_at`` and ``ended_at`` are epoch seconds on a process-wide clock that
anchors ``time.perf_counter`` to ``time.time`` once, at import. The clock is monotonic and has
sub-microsecond resolution (``time.time`` alone advances in 15.6 ms steps on Windows with
Python 3.12), so fast nodes and parallel ``Send`` workers keep their order, and
``ended_at - started_at`` matches ``duration_ms``. Use :func:`epoch_now` to take timestamps on
the same clock elsewhere, for example request start times in the load test.

Steps: every event also records the LangGraph step that ran the node (``TraceEvent.step``),
read from the run config while the node runs. The nodes of one step ran in parallel, so the UI
groups the ``Send`` workers of one round by their step; overlapping time windows would miss a
fast worker that finishes before the next one starts.

Importing the module loads only pydantic, not LangGraph (for :class:`TraceEvent` in a report
model, for example); the decorator imports LangGraph's config accessor when a node runs.
"""

import functools
import inspect
import logging
import time
from collections.abc import Callable, Mapping
from typing import Any, Final, Self, cast, overload

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "TRACE_KEY",
    "MetadataBuilder",
    "Summarizer",
    "TraceEvent",
    "epoch_now",
    "trace_events_from_chunk",
    "traced",
]

logger = logging.getLogger(__name__)

TRACE_KEY: Final = "trace"
"""State key that collects trace events in every graph of the project."""

type Summarizer = Callable[[dict[str, Any]], str]
"""Builds ``TraceEvent.summary`` from a node's own partial update."""

type MetadataBuilder = Callable[[dict[str, Any]], Mapping[str, Any]]
"""Builds ``TraceEvent.metadata`` from a node's own partial update."""

# Chunk entries that LangGraph adds next to the node entries of an "updates" chunk.
_NON_NODE_KEYS: Final = frozenset({"__interrupt__", "__metadata__"})

# One reading of each clock, taken together, maps perf_counter values to epoch seconds.
_PERF_ANCHOR: Final = time.perf_counter()
_EPOCH_ANCHOR: Final = time.time()


class TraceEvent(BaseModel):
    """One successful execution of a graph node.

    Events are immutable values. Derive a changed copy with
    ``TraceEvent.model_validate({**event.model_dump(), **changes})``, which validates it again;
    ``model_copy(update=...)`` would skip the validation.
    """

    model_config = ConfigDict(frozen=True)

    node: str = Field(min_length=1, description="Name of the node that produced the event.")
    started_at: float = Field(description="Start time in seconds since the Unix epoch.")
    ended_at: float = Field(description="End time in seconds since the Unix epoch.")
    duration_ms: float = Field(
        ge=0, description="Execution time in milliseconds, measured with time.perf_counter."
    )
    summary: str = Field(
        default="", description="One-line, human-readable outcome for the UI step panel."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Small JSON-serializable details, for example counts or scores.",
    )
    step: int | None = Field(
        default=None,
        description=(
            "LangGraph step (superstep) of the graph that ran the node, from the run config "
            "(`langgraph_step`): nodes of one step ran in parallel, such as the Send workers "
            "of one round. A subgraph counts its own steps. None outside a graph run."
        ),
    )

    @model_validator(mode="after")
    def _check_time_order(self) -> Self:
        """Reject events that end before they start."""
        if self.ended_at < self.started_at:
            raise ValueError("ended_at must not be earlier than started_at")
        return self


def epoch_now() -> float:
    """Return the current time on the trace clock.

    Returns:
        Seconds since the Unix epoch, monotonic and high-resolution within the process,
        directly comparable with ``TraceEvent.started_at`` and ``TraceEvent.ended_at``.
    """
    return _to_epoch(time.perf_counter())


@overload
def traced[F: Callable[..., Any]](fn: F, /) -> F: ...


@overload
def traced[F: Callable[..., Any]](
    *,
    node_name: str | None = None,
    summarize: Summarizer | None = None,
    metadata: MetadataBuilder | None = None,
) -> Callable[[F], F]: ...


def traced[F: Callable[..., Any]](
    fn: F | None = None,
    /,
    *,
    node_name: str | None = None,
    summarize: Summarizer | None = None,
    metadata: MetadataBuilder | None = None,
) -> F | Callable[[F], F]:
    """Record a :class:`TraceEvent` for every successful call of a LangGraph node.

    Use it bare (``@traced``) or with keyword arguments (``@traced()``,
    ``@traced(summarize=...)``), on plain sync functions and on ``async def`` functions. The
    wrapper keeps the wrapped function's name, docstring, signature and annotations
    (``functools.wraps``), so LangGraph still infers the node's input schema and injects the
    arguments the node declares, such as ``config`` and ``runtime``.

    The node returns a ``dict`` partial update or ``None``, and the wrapper returns a new dict
    with the event appended to the update's ``"trace"`` list (``{"trace": [event]}`` for
    ``None``). Events the node put there itself, the forwarded events of a subgraph, stay in
    front. Any other return value, ``Command`` included, raises ``TypeError``. If a graph's
    state has no ``trace`` key, LangGraph silently drops the extra key, so decorated nodes
    also work in graphs that do not collect traces. The event's ``step`` is the LangGraph
    step that ran the node, or ``None`` when the function is called outside a graph.

    Exceptions are never swallowed. When the node raises, including LangGraph control-flow
    exceptions such as ``GraphInterrupt``, the exception propagates unchanged and no event is
    recorded. Under a ``RetryPolicy`` each attempt is timed on its own and only the attempt
    that succeeds produces an event. Exceptions raised by ``summarize`` or ``metadata``
    propagate as well.

    Args:
        fn: The node function; given only by the bare ``@traced`` form.
        node_name: Name recorded in ``TraceEvent.node``; defaults to the function name. Keep
            it equal to the name the node is registered under in the graph (the default when
            node functions are named after their nodes), so that ``skip_forwarded`` of
            :func:`trace_events_from_chunk` recognises the node's own events.
        summarize: Builds ``TraceEvent.summary`` from the node's own update (``{}`` when the
            node returned ``None``). It runs after the timing stops, must be fast and pure,
            and must not mutate its argument.
        metadata: Builds ``TraceEvent.metadata`` (small, JSON-serializable values) from the
            node's own update, under the same rules as ``summarize``.

    Returns:
        The wrapped node function, or a decorator that produces it.

    Raises:
        TypeError: At decoration time, if the target is not callable, is a generator
            function, or has no ``__name__`` while ``node_name`` is not given. At call time,
            if the node returns anything but a dict or ``None``, or a ``"trace"`` value that
            is not a list of :class:`TraceEvent`.
    """
    if fn is not None:
        return _wrap(fn, name=node_name, summarize=summarize, metadata=metadata)

    def decorate(target: F) -> F:
        return _wrap(target, name=node_name, summarize=summarize, metadata=metadata)

    return decorate


def trace_events_from_chunk(chunk: object, *, skip_forwarded: bool = False) -> list[TraceEvent]:
    """Extract the trace events from one ``"updates"`` chunk of a graph stream.

    Accepts the three chunk shapes the project streams:

    - a ``version="v2"`` stream part, ``{"type": "updates", "ns": ..., "data": {node:
      update}}``, as the UI receives it with ``stream_mode=["updates", "values"]``; parts of
      other types yield no events;
    - ``{node: update}``, from ``graph.stream(..., stream_mode="updates")``;
    - ``(namespace, {node: update})``, from the same call with ``subgraphs=True``.

    Every other chunk yields no events, including the ``(mode, data)`` tuples of a
    ``version="v1"`` stream with several modes: stream several modes as ``version="v2"``
    parts. Within a chunk, the ``__interrupt__`` and ``__metadata__`` entries, nodes without
    an update, updates without a ``"trace"`` list and items that are not :class:`TraceEvent`
    instances are skipped.

    Args:
        chunk: One item yielded by ``graph.stream`` or ``graph.astream``.
        skip_forwarded: Keep only the events a node recorded itself, i.e. events whose
            ``node`` equals the node key of the update, and drop the events the node
            forwarded from a subgraph. Use it with ``subgraphs=True``, where the subgraph's
            own chunks have already delivered the forwarded events, and to show only the
            steps of the streamed graph itself (the UI step panel).

    Returns:
        The events in the order they appear in the chunk; empty when there are none.
    """
    events: list[TraceEvent] = []
    for node, update in _updates_of(chunk).items():
        if node in _NON_NODE_KEYS or not isinstance(update, Mapping):
            continue
        recorded = update.get(TRACE_KEY)
        if isinstance(recorded, list | tuple):
            events.extend(
                item
                for item in recorded
                if isinstance(item, TraceEvent) and (not skip_forwarded or item.node == node)
            )
    return events


def _to_epoch(perf: float) -> float:
    """Convert a ``time.perf_counter`` reading to seconds since the Unix epoch."""
    return _EPOCH_ANCHOR + (perf - _PERF_ANCHOR)


def _wrap[F: Callable[..., Any]](
    fn: F,
    *,
    name: str | None,
    summarize: Summarizer | None,
    metadata: MetadataBuilder | None,
) -> F:
    """Build the timing wrapper for one node function."""
    node = _node_name(fn, name)

    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_node(*args: Any, **kwargs: Any) -> dict[str, Any]:
            start = time.perf_counter()
            try:
                result = await fn(*args, **kwargs)
            except BaseException:
                _log_exit(node, start)
                raise
            return _finish(result, start, node, summarize, metadata)

        return cast(F, async_node)

    @functools.wraps(fn)
    def sync_node(*args: Any, **kwargs: Any) -> dict[str, Any]:
        start = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
        except BaseException:
            _log_exit(node, start)
            raise
        return _finish(result, start, node, summarize, metadata)

    return cast(F, sync_node)


def _node_name(fn: object, name: str | None) -> str:
    """Check that ``fn`` can be traced and return the node name its events record."""
    if not callable(fn):
        hint = "; pass a node name as traced(node_name=...)" if isinstance(fn, str) else ""
        raise TypeError(f"traced() expects a node function, got {type(fn).__name__}{hint}")
    if inspect.isgeneratorfunction(fn) or inspect.isasyncgenfunction(fn):
        raise TypeError("traced() supports plain sync and async node functions, not generators")
    node = name or getattr(fn, "__name__", None)
    if not node:
        raise TypeError("traced() needs node_name=... for a callable without __name__")
    return node


def _finish(
    result: object,
    start: float,
    node: str,
    summarize: Summarizer | None,
    metadata: MetadataBuilder | None,
) -> dict[str, Any]:
    """Stop the clock and return the node's update with its event appended."""
    end = time.perf_counter()
    update = _own_update(result, node)
    event = TraceEvent(
        node=node,
        started_at=_to_epoch(start),
        ended_at=_to_epoch(end),
        duration_ms=(end - start) * 1000.0,
        summary=summarize(update) if summarize else "",
        metadata=dict(metadata(update)) if metadata else {},
        step=_current_step(),
    )
    logger.debug("Node %r finished in %.1f ms", node, event.duration_ms)
    return {**update, TRACE_KEY: [*_forwarded_events(update, node), event]}


def _current_step() -> int | None:
    """Return the LangGraph step that is running the current node, or None outside a graph."""
    # Imported here and not at the top, so that importing this module loads no LangGraph. While
    # a graph runs a node, LangGraph is loaded already and the import costs nothing.
    from langgraph.config import get_config

    try:
        config = get_config()
    except RuntimeError:  # Called outside a runnable context: no graph is running the node.
        return None
    step = config.get("metadata", {}).get("langgraph_step")
    return step if isinstance(step, int) else None


def _log_exit(node: str, start: float) -> None:
    """Log that a node left with an exception; the caller re-raises it."""
    logger.debug(
        "Node %r raised after %.1f ms; no trace event recorded",
        node,
        (time.perf_counter() - start) * 1000.0,
    )


def _own_update(result: object, node: str) -> dict[str, Any]:
    """Return a node's result as its update: the dict itself, or ``{}`` for ``None``."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    raise TypeError(
        f"Node {node!r} returned {type(result).__name__}; a traced node must return a dict "
        "partial update or None (Command and other return values are not supported)"
    )


def _forwarded_events(update: dict[str, Any], node: str) -> list[TraceEvent]:
    """Return the events a node put under the trace key itself, after checking them."""
    value = update.get(TRACE_KEY, [])
    if isinstance(value, list | tuple):
        foreign = sorted(
            {type(item).__name__ for item in value if not isinstance(item, TraceEvent)}
        )
        if not foreign:
            return list(value)
        found = f"a {type(value).__name__} holding {', '.join(foreign)}"
    else:
        found = type(value).__name__
    raise TypeError(
        f"Node {node!r} returned {TRACE_KEY!r} as {found}; expected a list of TraceEvent"
    )


def _updates_of(chunk: object) -> Mapping[str, Any]:
    """Return the ``{node: update}`` mapping of an updates chunk, or ``{}`` for other chunks."""
    data: object = chunk
    if isinstance(chunk, Mapping) and {"type", "ns", "data"} <= chunk.keys():
        data = chunk["data"] if chunk["type"] == "updates" else None  # v2 stream part
    elif isinstance(chunk, tuple) and len(chunk) == 2 and isinstance(chunk[0], tuple):
        data = chunk[1]  # (namespace, {node: update}) with subgraphs=True
    return data if isinstance(data, Mapping) else {}
