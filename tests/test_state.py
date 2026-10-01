"""Tests for the state contracts in agentic_rag.rag.state and agentic_rag.agent.state.

The throwaway graphs below mirror the planned workflow (plan sections 5.1 and 5.2) with
trivial scripted nodes and exist only in this file. They prove that the contracts work in
real StateGraphs: the graphs start from their input schemas alone and return their output
schemas, the reducers accumulate as intended (messages, Send fan-in, trace), the re-plan
reset of subtask_results works, the verify loop follows the documented retry convention, and
the RAG subgraph's trace reaches the main trace.
"""

from collections import Counter
from collections.abc import Callable
from typing import Any, get_args

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Overwrite, Send
from pydantic import BaseModel, ValidationError

from agentic_rag.agent import state as agent_state
from agentic_rag.agent import types as agent_types
from agentic_rag.agent.state import (
    AgentInput,
    AgentOutput,
    AgentState,
    Intent,
    Subtask,
    SubtaskInput,
    SubtaskKind,
    SubtaskResult,
    Verdict,
)
from agentic_rag.rag.state import RagInput, RagOutput, RagState, Source
from agentic_rag.tracing import TRACE_KEY, TraceEvent, traced

RAG_NODES = ["rewrite_query", "retrieve", "grade_documents", "build_context"]
MAX_RETRIES = 2
DIRECT_REPLY = "Hello! Ask me about the knowledge base."
PARTIAL_NOTE = " (only partially answered)"


def _source(index: int = 1, **overrides: Any) -> Source:
    """Build a valid Source."""
    return Source(
        chunk_id=f"doc.md#{index}", source="doc.md", content=f"chunk {index}", **overrides
    )


RECORDS: list[BaseModel] = [
    _source(title="Doc", page=3, section="Intro", score=0.8),
    Subtask(id="s1", kind="retrieve", input="what is alpha"),
    Subtask(id="s2", kind="tool", input="add 2 and 3", tool_name="add", tool_args={"a": 2, "b": 3}),
    SubtaskResult(subtask_id="s1", kind="retrieve", output="[1] chunk 1", sources=[_source()]),
    SubtaskResult(subtask_id="s2", kind="tool", output="", ok=False, error="tool failed"),
    TraceEvent(
        node="retrieve", started_at=1.0, ended_at=1.25, duration_ms=250.0, metadata={"k": 4}
    ),
]


# --- records and types -----------------------------------------------------------------------


@pytest.mark.parametrize("record", RECORDS, ids=lambda record: type(record).__name__)
def test_records_round_trip_through_json(record: BaseModel) -> None:
    """Every record serializes to JSON and back without loss."""
    model = type(record)
    assert model.model_validate_json(record.model_dump_json()) == record
    assert model.model_validate(record.model_dump(mode="json")) == record


@pytest.mark.parametrize(
    ("model", "values"),
    [
        (Subtask, {"id": "s1", "kind": "search", "input": "x"}),
        (Subtask, {"id": "s1", "kind": "tool", "input": "x"}),
        (Subtask, {"id": "", "kind": "retrieve", "input": "x"}),
        (SubtaskResult, {"subtask_id": "s1", "kind": "tool", "output": "", "ok": False}),
        (SubtaskResult, {"subtask_id": "s1", "kind": "tool", "output": "4", "error": "boom"}),
        (Source, {"chunk_id": "c1", "source": "doc.md", "content": "x", "page": 0}),
        (Source, {"chunk_id": "", "source": "doc.md", "content": "x"}),
    ],
    ids=[
        "unknown-kind",
        "tool-without-name",
        "empty-id",
        "failure-without-error",
        "error-on-success",
        "page-not-1-based",
        "empty-chunk-id",
    ],
)
def test_records_reject_invalid_values(model: type[BaseModel], values: dict[str, Any]) -> None:
    """The records enforce their invariants at construction time."""
    with pytest.raises(ValidationError):
        model.model_validate(values)


def test_records_are_immutable_and_do_not_share_defaults() -> None:
    """Records are frozen values, and mutable defaults are per instance."""
    first = Subtask(id="s1", kind="retrieve", input="x")
    second = Subtask(id="s2", kind="retrieve", input="y")
    assert first.tool_args == {}
    assert first.tool_args is not second.tool_args
    result_a = SubtaskResult(subtask_id="s1", kind="retrieve", output="")
    result_b = SubtaskResult(subtask_id="s2", kind="retrieve", output="")
    assert result_a.sources is not result_b.sources
    with pytest.raises(ValidationError):
        first.input = "changed"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        _source().score = 1.0  # type: ignore[misc]
    assert first.model_copy(update={"input": "changed"}).input == "changed"


def test_subtask_json_schema_suits_structured_output() -> None:
    """The planner's output schema is flat, enumerated and fully described."""
    schema = Subtask.model_json_schema()
    assert set(schema["required"]) == {"id", "kind", "input"}
    assert schema["properties"]["kind"]["enum"] == ["retrieve", "tool"]
    assert all(prop.get("description") for prop in schema["properties"].values())
    assert "$defs" not in schema


def test_literal_types_and_schema_keys_match_the_plan() -> None:
    """Intents, verdicts, kinds and required keys are the ones the plan defines."""
    assert get_args(Intent) == ("direct", "single", "complex", "tool")
    assert get_args(Verdict) == ("grounded", "insufficient")
    assert get_args(SubtaskKind) == ("retrieve", "tool")
    assert AgentState.__required_keys__ == frozenset({"messages"})
    assert RagState.__required_keys__ == frozenset({"query"})
    assert set(SubtaskInput.__required_keys__) == {"subtask", "question"}
    for schema in (RagState, RagOutput, AgentState, AgentOutput):
        assert TRACE_KEY in schema.__annotations__


def test_state_reexports_the_light_literal_types() -> None:
    """agent.state re-exports the literal types of agent.types instead of defining its own."""
    for name in agent_types.__all__:
        assert getattr(agent_state, name) is getattr(agent_types, name)
        assert name in agent_state.__all__


# --- throwaway graphs that mirror the planned workflow ----------------------------------------


def _build_rag_graph() -> CompiledStateGraph:
    """Build a scripted RAG subgraph over RagState, RagInput and RagOutput."""

    @traced()
    def rewrite_query(state: RagState) -> dict[str, Any]:
        return {"rewritten_query": state["query"].strip().lower()}

    @traced()
    def retrieve(state: RagState) -> dict[str, Any]:
        query = state["rewritten_query"]
        documents = [
            Document(
                page_content=f"{query} fact {i}",
                metadata={"chunk_id": f"{query}#{i}", "source": "kb.md", "page": i},
            )
            for i in (1, 2, 3)
        ]
        return {"documents": documents, "scores": [0.9, 0.7, 0.1]}

    @traced()
    def grade_documents(state: RagState) -> dict[str, Any]:
        pairs = zip(state["documents"], state["scores"], strict=True)
        kept = [(document, score) for document, score in pairs if score >= 0.5]
        return {"documents": [d for d, _ in kept], "scores": [s for _, s in kept]}

    @traced()
    def build_context(state: RagState) -> dict[str, Any]:
        pairs = zip(state.get("documents", []), state.get("scores", []), strict=True)
        sources = [
            Source(
                chunk_id=document.metadata["chunk_id"],
                source=document.metadata["source"],
                content=document.page_content,
                page=document.metadata["page"],
                score=score,
            )
            for document, score in pairs
        ]
        context = "\n".join(f"[{i}] {s.content}" for i, s in enumerate(sources, start=1))
        return {"context": context, "sources": sources}

    builder = StateGraph(RagState, input_schema=RagInput, output_schema=RagOutput)
    for node in (rewrite_query, retrieve, grade_documents, build_context):
        builder.add_node(node)
    builder.add_edge(START, "rewrite_query")
    builder.add_edge("rewrite_query", "retrieve")
    builder.add_edge("retrieve", "grade_documents")
    builder.add_edge("grade_documents", "build_context")
    builder.add_edge("build_context", END)
    return builder.compile()


def _build_agent_graph(
    rag_graph: CompiledStateGraph,
    *,
    max_retries: int = MAX_RETRIES,
    insufficient_drafts: int = 1,
    reset_on_replan: bool = True,
    on_synthesis: Callable[[list[str]], None] = lambda ids: None,
) -> CompiledStateGraph:
    """Build a scripted main graph over AgentState, AgentInput and AgentOutput.

    The question prefix (``"complex: ..."``) selects the intent. The first
    ``insufficient_drafts`` drafts of a complex request are judged insufficient, so with the
    defaults the graph re-plans exactly once. ``retry_count`` follows the documented
    convention: it counts the re-plans, and the loop re-plans while it is below
    ``max_retries``.
    """

    @traced()
    def analyze_request(state: AgentState) -> dict[str, Any]:
        intent, _, question = str(state["messages"][-1].content).partition(":")
        question = question.strip()
        update: dict[str, Any] = {"question": question, "intent": intent}
        if intent == "direct":
            update["draft_answer"] = DIRECT_REPLY
        elif intent == "single":
            update["subtasks"] = [Subtask(id="s1", kind="retrieve", input=question)]
        elif intent == "tool":
            update["subtasks"] = [
                Subtask(
                    id="t1",
                    kind="tool",
                    input=question,
                    tool_name="add",
                    tool_args={"a": 1, "b": 1},
                )
            ]
        return update

    @traced()
    def plan_subtasks(state: AgentState) -> dict[str, Any]:
        # A re-plan follows an insufficient verdict, after retry_count earlier re-plans.
        replanning = state.get("verdict") == "insufficient"
        round_no = state.get("retry_count", 0) + 2 if replanning else 1
        subtasks = [
            Subtask(id=f"r{round_no}-alpha", kind="retrieve", input="Alpha"),
            Subtask(id=f"r{round_no}-beta", kind="retrieve", input="Beta"),
            Subtask(
                id=f"r{round_no}-sum",
                kind="tool",
                input="add 2 and 3",
                tool_name="add",
                tool_args={"a": 2, "b": 3},
            ),
        ]
        update: dict[str, Any] = {"subtasks": subtasks}
        if reset_on_replan:
            update["subtask_results"] = Overwrite([])
        return update

    @traced()
    def run_rag_subtask(state: SubtaskInput) -> dict[str, Any]:
        subtask = state["subtask"]
        rag: RagOutput = rag_graph.invoke(RagInput(query=subtask.input))
        result = SubtaskResult(
            subtask_id=subtask.id, kind="retrieve", output=rag["context"], sources=rag["sources"]
        )
        return {"subtask_results": [result], "trace": rag["trace"]}

    @traced()
    def call_tool(state: SubtaskInput) -> dict[str, Any]:
        subtask = state["subtask"]
        total = sum(subtask.tool_args.values())
        result = SubtaskResult(subtask_id=subtask.id, kind="tool", output=str(total))
        return {"subtask_results": [result]}

    @traced()
    def synthesize_answer(state: AgentState) -> dict[str, Any]:
        ids = sorted(result.subtask_id for result in state["subtask_results"])
        on_synthesis(ids)
        return {"draft_answer": f"Answer to {state['question']!r} from {len(ids)} results"}

    @traced()
    def verify_answer(state: AgentState) -> dict[str, Any]:
        # retry_count counts re-plans: 0 for the first draft, +1 for each re-planned draft.
        replanned = state.get("verdict") == "insufficient"
        retry_count = state.get("retry_count", 0) + 1 if replanned else 0
        rejected = state["intent"] == "complex" and retry_count < insufficient_drafts
        return {"verdict": "insufficient" if rejected else "grounded", "retry_count": retry_count}

    @traced()
    def finalize_response(state: AgentState) -> dict[str, Any]:
        answer = state["draft_answer"]
        if state.get("verdict") == "insufficient":
            answer += PARTIAL_NOTE
        # The citation contract: sub-task order, then rank order; a repeat keeps its place.
        numbered: dict[str, Source] = {}
        for result in state["subtask_results"]:
            for source in result.sources:
                numbered.setdefault(source.chunk_id, source)
        return {
            "answer": answer,
            "sources": list(numbered.values()),
            "messages": [AIMessage(content=answer)],
        }

    def dispatch_subtasks(state: AgentState) -> list[Send]:
        return [
            Send(
                "run_rag_subtask" if subtask.kind == "retrieve" else "call_tool",
                SubtaskInput(subtask=subtask, question=state["question"]),
            )
            for subtask in state["subtasks"]
        ]

    def route_after_analyze(state: AgentState) -> str | list[Send]:
        if state["intent"] == "direct":
            return "finalize_response"
        if state["intent"] == "complex":
            return "plan_subtasks"
        return dispatch_subtasks(state)  # the one-step plan of the single and tool routes

    def route_after_verify(state: AgentState) -> str:
        if state["verdict"] == "insufficient" and state["retry_count"] < max_retries:
            return "plan_subtasks"
        return "finalize_response"

    builder = StateGraph(AgentState, input_schema=AgentInput, output_schema=AgentOutput)
    for node in (
        analyze_request,
        plan_subtasks,
        run_rag_subtask,
        call_tool,
        synthesize_answer,
        verify_answer,
        finalize_response,
    ):
        builder.add_node(node)
    builder.add_edge(START, "analyze_request")
    builder.add_conditional_edges(
        "analyze_request",
        route_after_analyze,
        ["finalize_response", "plan_subtasks", "run_rag_subtask", "call_tool"],
    )
    builder.add_conditional_edges(
        "plan_subtasks", dispatch_subtasks, ["run_rag_subtask", "call_tool"]
    )
    builder.add_edge("run_rag_subtask", "synthesize_answer")
    builder.add_edge("call_tool", "synthesize_answer")
    builder.add_edge("synthesize_answer", "verify_answer")
    builder.add_conditional_edges(
        "verify_answer", route_after_verify, ["plan_subtasks", "finalize_response"]
    )
    builder.add_edge("finalize_response", END)
    return builder.compile()


def _round_ids(round_no: int) -> list[str]:
    """Return the sorted sub-task ids the scripted planner emits in a round."""
    return sorted(f"r{round_no}-{name}" for name in ("alpha", "beta", "sum"))


# --- RAG subgraph contract ---------------------------------------------------------------------


def test_rag_subgraph_starts_from_rag_input_and_returns_rag_output() -> None:
    """The subgraph needs only a query and returns exactly the RagOutput keys."""
    result = _build_rag_graph().invoke(RagInput(query="  Alpha "))
    assert set(result) == set(RagOutput.__annotations__)
    assert result["context"] == "[1] alpha fact 1\n[2] alpha fact 2"
    assert [source.chunk_id for source in result["sources"]] == ["alpha#1", "alpha#2"]
    assert all(isinstance(source, Source) for source in result["sources"])
    assert [event.node for event in result["trace"]] == RAG_NODES
    assert all(event.duration_ms >= 0 for event in result["trace"])


# --- main graph contract -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "expected_trace"),
    [
        ("direct: hello", ["analyze_request", "finalize_response"]),
        (
            "single: Alpha",
            [
                "analyze_request",
                *RAG_NODES,
                "run_rag_subtask",
                "synthesize_answer",
                "verify_answer",
                "finalize_response",
            ],
        ),
        (
            "tool: one plus one",
            [
                "analyze_request",
                "call_tool",
                "synthesize_answer",
                "verify_answer",
                "finalize_response",
            ],
        ),
    ],
    ids=["direct", "single", "tool"],
)
def test_agent_graph_starts_from_agent_input_and_returns_agent_output(
    question: str, expected_trace: list[str]
) -> None:
    """Every route starts from AgentInput alone and returns the AgentOutput keys."""
    graph = _build_agent_graph(_build_rag_graph())
    result = graph.invoke(AgentInput(messages=[HumanMessage(content=question)]))
    assert set(result) >= AgentOutput.__required_keys__
    assert set(result) <= set(AgentOutput.__annotations__)
    assert [event.node for event in result["trace"]] == expected_trace
    assert [type(message) for message in result["messages"]] == [HumanMessage, AIMessage]
    assert result["messages"][-1].content == result["answer"]


def test_direct_route_leaves_route_specific_keys_out() -> None:
    """Keys that depend on the route are absent when their nodes did not run."""
    graph = _build_agent_graph(_build_rag_graph())
    result = graph.invoke({"messages": [HumanMessage(content="direct: hello")]})
    assert result["answer"] == DIRECT_REPLY
    assert result["subtask_results"] == []
    assert result["sources"] == []
    assert {"verdict", "retry_count", "subtasks"}.isdisjoint(result)


def test_single_route_forwards_the_rag_trace_and_sources() -> None:
    """run_rag_subtask turns RagOutput into a SubtaskResult and forwards the RAG trace."""
    graph = _build_agent_graph(_build_rag_graph())
    result = graph.invoke({"messages": [("user", "single: Alpha")]})
    [subtask_result] = result["subtask_results"]
    assert [subtask.id for subtask in result["subtasks"]] == ["s1"]
    assert subtask_result.subtask_id == "s1"
    assert subtask_result.output == "[1] alpha fact 1\n[2] alpha fact 2"
    assert [source.chunk_id for source in result["sources"]] == ["alpha#1", "alpha#2"]
    assert result["verdict"] == "grounded"
    assert result["retry_count"] == 0


@pytest.mark.parametrize("reset_on_replan", [True, False], ids=["with-reset", "plain-append"])
def test_replanning_round_resets_subtask_results(reset_on_replan: bool) -> None:
    """Overwrite starts every planning round fresh; plain append keeps stale results."""
    rounds_seen: list[list[str]] = []
    graph = _build_agent_graph(
        _build_rag_graph(), reset_on_replan=reset_on_replan, on_synthesis=rounds_seen.append
    )
    result = graph.invoke({"messages": [HumanMessage(content="complex: compare alpha and beta")]})

    assert result["verdict"] == "grounded"
    assert result["retry_count"] == 1
    assert [subtask.id for subtask in result["subtasks"]] == ["r2-alpha", "r2-beta", "r2-sum"]
    if reset_on_replan:
        assert rounds_seen == [_round_ids(1), _round_ids(2)]
        assert sorted(r.subtask_id for r in result["subtask_results"]) == _round_ids(2)
    else:
        assert rounds_seen == [_round_ids(1), sorted(_round_ids(1) + _round_ids(2))]
        assert len(result["subtask_results"]) == 6


@pytest.mark.parametrize("max_retries", [0, MAX_RETRIES])
def test_verify_loop_replans_at_most_max_retries_times(max_retries: int) -> None:
    """With every draft rejected, the run re-plans max_retries times and ends partial."""
    graph = _build_agent_graph(
        _build_rag_graph(), max_retries=max_retries, insufficient_drafts=max_retries + 99
    )
    result = graph.invoke({"messages": [HumanMessage(content="complex: compare alpha and beta")]})

    runs = Counter(event.node for event in result["trace"])
    assert runs["plan_subtasks"] == runs["verify_answer"] == max_retries + 1
    assert result["verdict"] == "insufficient"
    assert result["retry_count"] == max_retries
    assert result["answer"].endswith(PARTIAL_NOTE)
    assert sorted(subtask.id for subtask in result["subtasks"]) == _round_ids(max_retries + 1)


def test_complex_route_fans_out_and_traces_every_step() -> None:
    """Each round fans out three Send workers whose results and traces fan back in."""
    graph = _build_agent_graph(_build_rag_graph())
    result = graph.invoke({"messages": [HumanMessage(content="complex: compare alpha and beta")]})

    trace = result["trace"]
    assert Counter(event.node for event in trace) == {
        "analyze_request": 1,
        "plan_subtasks": 2,
        "run_rag_subtask": 4,
        "call_tool": 2,
        **dict.fromkeys(RAG_NODES, 4),
        "synthesize_answer": 2,
        "verify_answer": 2,
        "finalize_response": 1,
    }
    assert trace[0].node == "analyze_request"
    assert trace[-1].node == "finalize_response"
    assert all(event.duration_ms >= 0 for event in trace)
    assert [r.subtask_id for r in result["subtask_results"]] == ["r2-alpha", "r2-beta", "r2-sum"]
    # The global citation numbering: sub-task (plan) order, then rank order.
    assert [source.chunk_id for source in result["sources"]] == [
        "alpha#1",
        "alpha#2",
        "beta#1",
        "beta#2",
    ]
    assert [type(message) for message in result["messages"]] == [HumanMessage, AIMessage]


def test_reducer_keys_start_empty_and_other_keys_are_absent() -> None:
    """Nodes see the reducer keys as empty lists and must use get() for the others."""
    seen: dict[str, Any] = {}

    def probe(state: AgentState) -> dict[str, Any]:
        seen.update(state)
        return {}

    graph = (
        StateGraph(AgentState, input_schema=AgentInput, output_schema=AgentOutput)
        .add_node(probe)
        .add_edge(START, "probe")
        .compile()
    )
    graph.invoke({"messages": [HumanMessage(content="hi")]})
    assert set(seen) == {"messages", "subtask_results", "trace"}
    assert seen["subtask_results"] == []
    assert seen["trace"] == []
