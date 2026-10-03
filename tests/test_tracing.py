"""Tests for the step-trace primitives in agentic_rag.tracing.

The graphs here are tiny throwaway StateGraphs that exist only in this file. They check that
traced nodes keep working inside real LangGraph graphs (schema inference, argument injection,
partial-bound dependencies, Send fan-out, subgraph forwarding, async execution) and that
trace_events_from_chunk understands the three "updates" chunk shapes the project streams.
"""

import asyncio
import functools
import inspect
import itertools
import operator
import time
import typing
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

import pytest
from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, Overwrite, Send, StreamWriter
from pydantic import ValidationError

from agentic_rag.tracing import TRACE_KEY, TraceEvent, epoch_now, trace_events_from_chunk, traced


class FlowState(TypedDict, total=False):
    """State of the throwaway test graphs."""

    text: str
    items: Annotated[list[str], operator.add]
    trace: Annotated[list[TraceEvent], operator.add]


class ItemInput(TypedDict):
    """Send payload of the throwaway worker nodes."""

    item: str


@dataclass
class Ctx:
    """Runtime context of the injection test."""

    user: str


class _BoomError(Exception):
    """Raised by failing test nodes."""


def _event(node: str = "n", **overrides: Any) -> TraceEvent:
    """Build a valid event with fixed timestamps."""
    values: dict[str, Any] = {
        "node": node,
        "started_at": 100.0,
        "ended_at": 100.5,
        "duration_ms": 500.0,
    }
    return TraceEvent(**(values | overrides))


def _names(events: Iterable[TraceEvent]) -> list[str]:
    """Return the node names of events, in order."""
    return [event.node for event in events]


async def _sleep_at_least(seconds: float) -> None:
    """Await until at least ``seconds`` of ``time.perf_counter`` time have passed.

    A plain ``asyncio.sleep`` can end early on Windows: a wait shorter than the system timer
    tick (15.6 ms by default) may time out early, and the event loop, which schedules timers on
    ``time.monotonic``, fires a timer up to one tick before it is due. ``traced`` measures with
    ``perf_counter``, so the duration assertions need this guarantee.
    """
    deadline = time.perf_counter() + seconds
    while (remaining := deadline - time.perf_counter()) > 0:
        await asyncio.sleep(remaining)


# --- TraceEvent and the clock ------------------------------------------------------------


def test_trace_event_round_trips_through_json() -> None:
    """A TraceEvent serializes to JSON and back without loss."""
    event = _event(summary="2 chunks kept", metadata={"k": 4, "ids": ["a", "b"]})
    assert TraceEvent.model_validate_json(event.model_dump_json()) == event
    assert TraceEvent.model_validate(event.model_dump(mode="json")) == event


def test_trace_event_defaults_are_empty_and_independent() -> None:
    """Summary and metadata default to empty values that instances do not share."""
    first, second = _event(), _event()
    assert first.summary == ""
    assert first.metadata == {}
    assert first.metadata is not second.metadata


@pytest.mark.parametrize(
    "overrides",
    [{"duration_ms": -1.0}, {"ended_at": 99.0}, {"node": ""}],
    ids=["negative-duration", "ends-before-start", "empty-node"],
)
def test_trace_event_rejects_invalid_values(overrides: dict[str, Any]) -> None:
    """Negative durations, reversed timestamps and empty node names are rejected."""
    with pytest.raises(ValidationError):
        _event(**overrides)


def test_trace_event_is_immutable() -> None:
    """Events are frozen; the documented copy carries a change and still validates it."""
    event = _event()
    with pytest.raises(ValidationError):
        event.node = "other"  # type: ignore[misc]
    changed = TraceEvent.model_validate({**event.model_dump(), "summary": "changed"})
    assert changed.summary == "changed"
    assert changed.node == event.node
    with pytest.raises(ValidationError):
        TraceEvent.model_validate({**event.model_dump(), "ended_at": 0.0})


def test_epoch_now_is_monotonic_epoch_time() -> None:
    """The trace clock runs in epoch seconds, close to time.time, and never goes back."""
    readings = [epoch_now() for _ in range(1000)]
    assert readings == sorted(readings)
    assert abs(readings[-1] - time.time()) < 1.0


# --- traced: wrapper identity and naming -------------------------------------------------


def _sample_node(state: FlowState, config: RunnableConfig) -> dict[str, Any]:
    """Return a fixed update (sample docstring)."""
    return {"text": "x"}


def test_traced_preserves_name_docstring_signature_and_annotations() -> None:
    """The wrapper looks like the wrapped function to LangGraph and to readers."""
    wrapped = traced()(_sample_node)
    assert wrapped.__name__ == "_sample_node"
    assert wrapped.__qualname__ == _sample_node.__qualname__
    assert wrapped.__doc__ == _sample_node.__doc__
    assert wrapped.__module__ == _sample_node.__module__
    assert wrapped.__wrapped__ is _sample_node  # type: ignore[attr-defined]
    assert inspect.signature(wrapped) == inspect.signature(_sample_node)
    assert typing.get_type_hints(wrapped) == typing.get_type_hints(_sample_node)


def test_traced_node_name_defaults_to_the_function_name() -> None:
    """The bare and keyword forms record the function name unless node_name is given."""

    def plain(state: FlowState) -> dict[str, Any]:
        return {}

    assert _names(traced(plain)({})[TRACE_KEY]) == ["plain"]
    assert _names(traced()(plain)({})[TRACE_KEY]) == ["plain"]
    assert _names(traced(node_name="custom")(plain)({})[TRACE_KEY]) == ["custom"]
    assert _names(traced(node_name="named")(functools.partial(plain))({})[TRACE_KEY]) == ["named"]


def test_traced_rejects_targets_it_cannot_wrap() -> None:
    """Generators, non-callables, nameless callables and a positional name are rejected."""

    def generator(state: FlowState) -> Iterable[dict[str, Any]]:
        yield {}

    async def async_generator(state: FlowState) -> typing.AsyncIterator[dict[str, Any]]:
        yield {}

    def plain(state: FlowState) -> dict[str, Any]:
        return {}

    for target in (generator, async_generator, 42, functools.partial(plain)):
        with pytest.raises(TypeError):
            traced()(target)  # type: ignore[type-var]
    with pytest.raises(TypeError, match=r"node_name=\.\.\."):
        traced("custom")  # type: ignore[call-overload]


# --- traced: what the wrapper returns ----------------------------------------------------


def test_traced_appends_the_event_to_a_new_dict() -> None:
    """A dict update gets the event; the node's own dict is not mutated."""
    own_update = {"text": "x"}

    @traced()
    def node(state: FlowState) -> dict[str, Any]:
        return own_update

    before = epoch_now()
    update = node({})
    after = epoch_now()
    assert own_update == {"text": "x"}
    assert update["text"] == "x"
    [event] = update[TRACE_KEY]
    assert event.node == "node"
    assert before <= event.started_at <= event.ended_at <= after
    assert event.duration_ms >= 0
    assert event.step is None  # Called outside a graph.


def test_traced_measures_the_duration_with_perf_counter() -> None:
    """duration_ms covers the node body, and the timestamps agree with it."""

    @traced()
    def slow(state: FlowState) -> dict[str, Any]:
        time.sleep(0.02)
        return {}

    [event] = slow({})[TRACE_KEY]
    assert event.duration_ms >= 15
    assert event.ended_at - event.started_at == pytest.approx(event.duration_ms / 1000, abs=1e-5)


def test_traced_records_a_step_for_a_node_without_update() -> None:
    """A node that returns None still produces exactly one event."""

    @traced()
    def noop(state: FlowState) -> None:
        return None

    update = noop({})
    assert list(update) == [TRACE_KEY]
    assert _names(update[TRACE_KEY]) == ["noop"]


def test_traced_keeps_events_the_node_returned_in_front() -> None:
    """Forwarded subgraph events stay in front of the node's own event."""
    forwarded = [_event("retrieve"), _event("build_context")]

    @traced()
    def parent(state: FlowState) -> dict[str, Any]:
        return {"trace": forwarded}

    assert _names(parent({})[TRACE_KEY]) == ["retrieve", "build_context", "parent"]
    assert len(forwarded) == 2


@pytest.mark.parametrize(
    ("value", "found"),
    [
        (_event(), "as TraceEvent"),
        (Overwrite([]), "as Overwrite"),
        ([_event(), {"node": "x"}], "as a list holding dict"),
        ("retrieve", "as str"),
    ],
    ids=["bare-event", "overwrite", "foreign-item", "string"],
)
def test_traced_rejects_a_trace_value_that_is_not_a_list_of_events(
    value: object, found: str
) -> None:
    """A malformed trace value, an Overwrite reset included, fails loudly at the node."""

    @traced()
    def malformed(state: FlowState) -> dict[str, Any]:
        return {"trace": value}

    with pytest.raises(TypeError, match=rf"'malformed' returned 'trace' {found}; expected a list"):
        malformed({})


@pytest.mark.parametrize(
    "result",
    [
        Command(goto="next"),
        Command(update={"text": "x"}),
        [Command(update={"text": "x"})],
        [("text", "x")],
        _event(),
        "text",
    ],
    ids=["command", "command-update", "command-list", "update-pairs", "model", "string"],
)
def test_traced_rejects_results_other_than_a_dict_or_none(result: object) -> None:
    """Command and every other non-dict result raise TypeError, in sync and async nodes."""

    @traced()
    def odd(state: FlowState) -> Any:
        return result

    @traced()
    async def odd_async(state: FlowState) -> Any:
        return result

    expected = rf"'odd(_async)?' returned {type(result).__name__}; a traced node must return a dict"
    with pytest.raises(TypeError, match=expected):
        odd({})
    with pytest.raises(TypeError, match=expected):
        asyncio.run(odd_async({}))


def test_traced_callbacks_receive_the_node_update() -> None:
    """summarize and metadata see the node's own update and fill the event."""
    seen: list[dict[str, Any]] = []

    def summarize(update: dict[str, Any]) -> str:
        seen.append(update)
        return f"{len(update['items'])} items"

    @traced(summarize=summarize, metadata=lambda update: {"count": len(update["items"])})
    def collect(state: FlowState) -> dict[str, Any]:
        return {"items": ["a", "b"]}

    @traced(summarize=lambda update: ",".join(sorted(update)) or "no update")
    def nothing(state: FlowState) -> None:
        return None

    [event] = collect({})[TRACE_KEY]
    assert event.summary == "2 items"
    assert event.metadata == {"count": 2}
    assert seen == [{"items": ["a", "b"]}]
    assert nothing({})[TRACE_KEY][0].summary == "no update"


def test_traced_propagates_exceptions_unchanged() -> None:
    """Node and callback exceptions reach the caller as the same objects."""
    sync_error, async_error = _BoomError("sync"), _BoomError("async")

    @traced()
    def failing(state: FlowState) -> dict[str, Any]:
        raise sync_error

    @traced()
    async def failing_async(state: FlowState) -> dict[str, Any]:
        raise async_error

    @traced(summarize=lambda update: update["missing"])
    def bad_summary(state: FlowState) -> dict[str, Any]:
        return {}

    with pytest.raises(_BoomError) as sync_info:
        failing({})
    assert sync_info.value is sync_error
    with pytest.raises(_BoomError) as async_info:
        asyncio.run(failing_async({}))
    assert async_info.value is async_error
    with pytest.raises(KeyError):
        bad_summary({})


def test_traced_async_node_stays_a_coroutine_function() -> None:
    """Async nodes get an async wrapper, so LangGraph awaits them natively."""

    @traced()
    async def async_node(state: FlowState) -> dict[str, Any]:
        await _sleep_at_least(0.01)
        return {"text": "done"}

    assert inspect.iscoroutinefunction(async_node)
    update = asyncio.run(async_node({}))
    assert update["text"] == "done"
    [event] = update[TRACE_KEY]
    assert event.node == "async_node"
    assert event.duration_ms >= 5


# --- traced inside real graphs ------------------------------------------------------------


def test_traced_nodes_record_their_steps_in_a_graph() -> None:
    """Sequential traced nodes leave ordered, non-overlapping events in the final state."""

    @traced()
    def first(state: FlowState) -> dict[str, Any]:
        return {"text": "a"}

    @traced()
    def second(state: FlowState) -> None:
        return None

    @traced()
    def third(state: FlowState) -> dict[str, Any]:
        return {"text": state["text"] + "c"}

    graph = (
        StateGraph(FlowState)
        .add_node(first)
        .add_node(second)
        .add_node(third)
        .add_edge(START, "first")
        .add_edge("first", "second")
        .add_edge("second", "third")
        .add_edge("third", END)
        .compile()
    )
    result = graph.invoke({"text": ""})
    events = result["trace"]
    assert result["text"] == "ac"
    assert _names(events) == ["first", "second", "third"]
    assert [event.step for event in events] == [1, 2, 3]  # LangGraph's langgraph_step
    assert all(event.duration_ms >= 0 for event in events)
    assert all(a.ended_at <= b.started_at for a, b in itertools.pairwise(events))


def test_traced_keeps_schema_inference_injection_and_partial_binding() -> None:
    """LangGraph still infers input schemas and injects arguments; partials keep the name."""

    @traced()
    def configured(
        state: FlowState, config: RunnableConfig, runtime: Runtime[Ctx]
    ) -> dict[str, Any]:
        node = config["metadata"]["langgraph_node"]
        return {"text": f"{config['configurable']['tag']}/{runtime.context.user}/{node}"}

    @traced()
    def worker(state: ItemInput) -> dict[str, Any]:
        return {"items": [state["item"]]}

    @traced()
    def bound_worker(state: ItemInput, *, suffix: str) -> dict[str, Any]:
        return {"items": [state["item"] + suffix]}

    builder = StateGraph(FlowState, context_schema=Ctx)
    builder.add_node(configured)
    builder.add_node(worker)
    # The project's convention: the dependency is bound with a partial and the node registered
    # under its name; the partial hides the payload type, so the input schema is explicit.
    builder.add_node(
        "bound_worker", functools.partial(bound_worker, suffix="!"), input_schema=ItemInput
    )
    builder.add_edge(START, "configured")
    builder.add_conditional_edges(
        "configured",
        lambda state: [Send("worker", {"item": "x"}), Send("bound_worker", {"item": "y"})],
        ["worker", "bound_worker"],
    )
    builder.add_edge("worker", END)
    builder.add_edge("bound_worker", END)
    assert builder.nodes["worker"].input_schema is ItemInput

    graph = builder.compile()
    result = graph.invoke({"text": ""}, {"configurable": {"tag": "t1"}}, context=Ctx(user="u1"))
    assert result["text"] == "t1/u1/configured"
    assert sorted(result["items"]) == ["x", "y!"]
    assert Counter(_names(result["trace"])) == {"configured": 1, "worker": 1, "bound_worker": 1}
    # The Send workers of one round share the LangGraph step after the one that sent them.
    steps = {event.node: event.step for event in result["trace"]}
    assert steps == {"configured": 1, "worker": 2, "bound_worker": 2}


def test_traced_nodes_work_in_graphs_without_a_trace_key() -> None:
    """LangGraph drops the trace key when the state does not declare it."""

    class PlainState(TypedDict):
        text: str

    @traced()
    def shout(state: PlainState) -> dict[str, Any]:
        return {"text": state["text"].upper()}

    graph = StateGraph(PlainState).add_node(shout).add_edge(START, "shout").compile()
    assert graph.invoke({"text": "hi"}) == {"text": "HI"}


def test_node_exceptions_propagate_out_of_the_graph() -> None:
    """A failing traced node fails the run with its own exception."""
    error = _BoomError("node failed")

    @traced()
    def explode(state: FlowState) -> dict[str, Any]:
        raise error

    graph = StateGraph(FlowState).add_node(explode).add_edge(START, "explode").compile()
    with pytest.raises(_BoomError) as info:
        graph.invoke({"text": ""})
    assert info.value is error


def test_async_nodes_are_traced_in_a_graph() -> None:
    """ainvoke and astream record async nodes, and parallel Send workers overlap in time."""

    @traced()
    async def begin(state: FlowState) -> dict[str, Any]:
        return {"text": "go"}

    @traced()
    async def slow_worker(state: ItemInput) -> dict[str, Any]:
        await _sleep_at_least(0.03)
        return {"items": [state["item"]]}

    @traced()
    def finish(state: FlowState) -> dict[str, Any]:
        return {"text": "done"}

    builder = StateGraph(FlowState)
    for node in (begin, slow_worker, finish):
        builder.add_node(node)
    builder.add_edge(START, "begin")
    builder.add_conditional_edges(
        "begin", lambda state: [Send("slow_worker", {"item": i}) for i in "abc"], ["slow_worker"]
    )
    builder.add_edge("slow_worker", "finish")
    builder.add_edge("finish", END)
    graph = builder.compile()

    async def run() -> tuple[dict[str, Any], list[TraceEvent]]:
        result = await graph.ainvoke({"text": ""})
        streamed = [
            event
            async for chunk in graph.astream({"text": ""}, stream_mode="updates")
            for event in trace_events_from_chunk(chunk)
        ]
        return result, streamed

    result, streamed = asyncio.run(run())
    expected = {"begin": 1, "slow_worker": 3, "finish": 1}
    assert sorted(result["items"]) == ["a", "b", "c"]
    assert Counter(_names(result["trace"])) == expected
    assert Counter(_names(streamed)) == expected
    workers = [event for event in result["trace"] if event.node == "slow_worker"]
    assert all(event.duration_ms >= 20 for event in workers)
    assert max(event.started_at for event in workers) < min(event.ended_at for event in workers)


# --- trace_events_from_chunk ---------------------------------------------------------------

EXPECTED_PARENT_TRACE = {"plan": 1, "child_step": 2, "worker": 2, "finish": 1}


def _parent_graph() -> CompiledStateGraph:
    """Build a graph whose Send workers invoke a child subgraph and forward its trace."""

    @traced()
    def child_step(state: FlowState) -> dict[str, Any]:
        return {"text": state["text"].upper()}

    child = StateGraph(FlowState).add_node(child_step).add_edge(START, "child_step").compile()

    @traced()
    def plan(state: FlowState, writer: StreamWriter) -> dict[str, Any]:
        writer({"progress": "planned"})
        return {"text": "planned"}

    @traced()
    def worker(state: ItemInput) -> dict[str, Any]:
        out = child.invoke({"text": state["item"]})
        return {"items": [out["text"]], "trace": out["trace"]}

    @traced()
    def finish(state: FlowState) -> dict[str, Any]:
        return {"text": ",".join(sorted(state["items"]))}

    builder = StateGraph(FlowState)
    for node in (plan, worker, finish):
        builder.add_node(node)
    builder.add_edge(START, "plan")
    builder.add_conditional_edges(
        "plan", lambda state: [Send("worker", {"item": i}) for i in ("a", "b")], ["worker"]
    )
    builder.add_edge("worker", "finish")
    builder.add_edge("finish", END)
    return builder.compile()


def test_updates_stream_without_subgraphs() -> None:
    """Each chunk carries its node's events; forwarded child events arrive with the worker."""
    graph = _parent_graph()
    per_chunk = [
        _names(trace_events_from_chunk(chunk))
        for chunk in graph.stream({"text": ""}, stream_mode="updates")
    ]
    assert per_chunk == [
        ["plan"],
        ["child_step", "worker"],
        ["child_step", "worker"],
        ["finish"],
    ]


def test_v2_updates_parts_add_up_to_the_final_trace() -> None:
    """Streamed as the UI streams, the updates parts carry exactly the final state's trace."""
    graph = _parent_graph()
    streamed: list[TraceEvent] = []
    final: dict[str, Any] = {}
    for part in graph.stream({"text": ""}, stream_mode=["updates", "values"], version="v2"):
        if part["type"] == "values":
            final = part["data"]
        streamed.extend(trace_events_from_chunk(part))
    assert Counter(e.model_dump_json() for e in streamed) == Counter(
        e.model_dump_json() for e in final["trace"]
    )


def test_updates_stream_with_subgraphs() -> None:
    """Subgraph chunks deliver child events live; skip_forwarded removes the repeats."""
    graph = _parent_graph()
    chunks = list(graph.stream({"text": ""}, stream_mode="updates", subgraphs=True))
    child_chunks = [chunk for chunk in chunks if chunk[0]]
    assert len(child_chunks) == 2
    assert all(namespace[0].startswith("worker:") for namespace, _ in child_chunks)
    assert [_names(trace_events_from_chunk(chunk)) for chunk in child_chunks] == [
        ["child_step"],
        ["child_step"],
    ]

    everything = [event for chunk in chunks for event in trace_events_from_chunk(chunk)]
    once = [
        event for chunk in chunks for event in trace_events_from_chunk(chunk, skip_forwarded=True)
    ]
    assert Counter(_names(everything)) == {"plan": 1, "child_step": 4, "worker": 2, "finish": 1}
    assert Counter(_names(once)) == EXPECTED_PARENT_TRACE


def test_skip_forwarded_matches_the_final_trace_of_the_same_run() -> None:
    """With subgraphs=True and skip_forwarded, every event of the run arrives exactly once."""
    graph = _parent_graph()
    once: list[TraceEvent] = []
    final: dict[str, Any] = {}
    for part in graph.stream(
        {"text": ""}, stream_mode=["updates", "values"], subgraphs=True, version="v2"
    ):
        if part["type"] == "values" and not part["ns"]:
            final = part["data"]
        once.extend(trace_events_from_chunk(part, skip_forwarded=True))
    assert Counter(e.model_dump_json() for e in once) == Counter(
        e.model_dump_json() for e in final["trace"]
    )


@pytest.mark.parametrize(
    "stream_kwargs",
    [
        {"stream_mode": "updates"},
        {"stream_mode": "updates", "subgraphs": True},
        {"stream_mode": "updates", "version": "v2"},
        {"stream_mode": ["updates", "custom"], "subgraphs": True, "version": "v2"},
    ],
    ids=["mapping", "namespace-tuple", "v2", "v2-multi-subgraphs"],
)
def test_trace_events_from_every_supported_chunk_shape(stream_kwargs: dict[str, Any]) -> None:
    """Every supported shape yields the run's events; custom parts yield none."""
    graph = _parent_graph()
    skip_forwarded = bool(stream_kwargs.get("subgraphs"))
    events: list[TraceEvent] = []
    custom_parts = 0
    for chunk in graph.stream({"text": ""}, **stream_kwargs):
        found = trace_events_from_chunk(chunk, skip_forwarded=skip_forwarded)
        if isinstance(chunk, dict) and chunk.get("type") == "custom":
            custom_parts += 1
            assert found == []
        events.extend(found)
    assert Counter(_names(events)) == EXPECTED_PARENT_TRACE
    assert custom_parts == (1 if isinstance(stream_kwargs["stream_mode"], list) else 0)


def test_trace_events_from_chunk_ignores_everything_else() -> None:
    """Other chunks, reserved entries, foreign values and v1 multi-mode tuples yield nothing."""
    event = _event("n")
    assert trace_events_from_chunk({"n": {"trace": [event]}}) == [event]
    assert trace_events_from_chunk((("sub:1",), {"n": {"trace": [event]}})) == [event]
    assert trace_events_from_chunk({"n": {"trace": [event]}, "__metadata__": {"cached": True}}) == [
        event
    ]
    assert trace_events_from_chunk({"n": {"trace": [event, "junk", {"node": "x"}]}}) == [event]
    assert trace_events_from_chunk({"m": {"trace": [event]}}, skip_forwarded=True) == []

    no_events: list[Any] = [
        {"n": None},
        {"n": {"text": "x"}},
        {"n": {"trace": "not a list"}},
        {"n": [{"trace": [event]}]},
        {"__interrupt__": ()},
        {"__metadata__": {"trace": [event]}},
        {"text": "a values chunk", "trace": [event]},
        {"type": "values", "ns": (), "data": {"n": {"trace": [event]}}},
        {"type": "custom", "ns": (), "data": {"n": {"trace": [event]}}},
        ("updates", {"n": {"trace": [event]}}),
        ((), "updates", {"n": {"trace": [event]}}),
        (AIMessageChunk(content="token"), {"langgraph_node": "n"}),
        None,
        42,
        "updates",
        [],
        (),
    ]
    for chunk in no_events:
        assert trace_events_from_chunk(chunk) == [], chunk
