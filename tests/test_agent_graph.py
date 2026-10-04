"""Tests for the main agentic workflow: its contract, the routing, each node and the graph.

The contract part pins what plan section 5.1 fixes: the node names, the node signatures with
their explicit dependencies and the routing annotations. The nodes are then called directly
with scripted chat models (``ScriptedChatModel``) and stand-in tools, and the compiled graph
runs offline in fake mode against a small index under ``tmp_path``: every route, the parallel
fan-out, the bounded re-plan loop and the citation numbering. A last group checks that the
rules of ``DEFAULT_FAKE_RULES`` still recognise the real prompts.
"""

import inspect
import types
import typing
from collections.abc import Callable, Mapping
from typing import Annotated, Any, Literal, get_args, get_origin, get_type_hints

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool, ToolException
from langgraph.types import Overwrite, Send

from agentic_rag import cli
from agentic_rag.agent import graph, nodes, routing, tools
from agentic_rag.agent.nodes import MAX_SUBTASKS, NO_DRAFT, PARTIAL_NOTE, numbered_sources
from agentic_rag.agent.prompts import (
    ANALYZE_INSTRUCTIONS,
    PLAN_INSTRUCTIONS,
    SYNTHESIZE_INSTRUCTIONS,
    VERIFY_INSTRUCTIONS,
    analyze_messages,
    format_material,
    plan_messages,
    synthesize_messages,
    verify_messages,
)
from agentic_rag.agent.state import AgentState, Subtask, SubtaskInput, SubtaskResult
from agentic_rag.agent.tools import (
    CHECK_CONTRAST,
    CheckContrastTool,
    CssSpecificityTool,
    SearchKnowledgeBaseTool,
)
from agentic_rag.config import Settings
from agentic_rag.ingestion.index import build_index
from agentic_rag.llm import DEFAULT_FAKE_RULES, FakeRule, ScriptedChatModel, render_prompt
from agentic_rag.rag.state import RagInput, RagOutput, Source
from agentic_rag.tracing import TraceEvent

PLAN_NODE_NAMES = (
    "analyze_request",
    "plan_subtasks",
    "run_rag_subtask",
    "call_tool",
    "synthesize_answer",
    "verify_answer",
    "finalize_response",
)
SEND_WORKERS = frozenset({"run_rag_subtask", "call_tool"})

# The routing rules of plan section 5.1: (nodes returned by name, nodes reached through Send).
PLANNED_ROUTES: dict[str, tuple[set[str], set[str]]] = {
    "route_after_analyze": (
        {"finalize_response", "plan_subtasks"},
        {"run_rag_subtask", "call_tool"},
    ),
    "dispatch_subtasks": (set(), {"run_rag_subtask", "call_tool"}),
    "route_after_verify": ({"plan_subtasks", "finalize_response"}, set()),
}

# What build_agent_graph binds to each node; it mirrors the node kinds of plan section 5.1.
NODE_DEPENDENCIES: dict[str, set[str]] = {
    "analyze_request": {"chat_model", "tools"},
    "plan_subtasks": {"chat_model", "tools"},
    "run_rag_subtask": {"search_tool"},
    "call_tool": {"tools"},
    "synthesize_answer": {"chat_model"},
    "verify_answer": {"chat_model"},
    "finalize_response": set(),
}
DEPENDENCY_TYPES: dict[str, Any] = {
    "chat_model": BaseChatModel,
    "tools": Mapping[str, BaseTool],
    "search_tool": SearchKnowledgeBaseTool,
}

TOOLS: dict[str, BaseTool] = {
    tool.name: tool for tool in (CheckContrastTool(), CssSpecificityTool())
}


def scripted(*rules: tuple[str, str]) -> ScriptedChatModel:
    """A scripted chat model with ``(pattern, reply)`` rules."""
    return ScriptedChatModel(rules=[FakeRule(pattern=p, reply=r) for p, r in rules])


def own_update(update: dict[str, Any]) -> dict[str, Any]:
    """A node's update without the trace event that ``traced`` adds."""
    return {key: value for key, value in update.items() if key != "trace"}


def source(chunk_id: str, text: str = "Text.") -> Source:
    """A retrieved chunk."""
    return Source(chunk_id=chunk_id, source=f"{chunk_id}.md", content=f"Title\n\n{text}")


def retrieved(subtask_id: str, *sources: Source) -> SubtaskResult:
    """A successful retrieve result."""
    return SubtaskResult(
        subtask_id=subtask_id, kind="retrieve", output="[1] ...", sources=list(sources)
    )


def user(text: str) -> AgentState:
    """A state with one user message."""
    return AgentState(messages=[HumanMessage(content=text)])


# --- the contract ---------------------------------------------------------------------------


def _union_members(annotation: Any) -> tuple[Any, ...]:
    """Return the members of a union annotation, or the annotation alone."""
    if get_origin(annotation) in (typing.Union, types.UnionType):
        return get_args(annotation)
    return (annotation,)


def _literal_names(annotation: Any) -> set[str]:
    """Return the string values of a Literal annotation; empty for anything else."""
    if get_origin(annotation) is not Literal:
        return set()
    return set(get_args(annotation))


def _route_targets(route: Callable[..., Any]) -> tuple[set[str], set[str]]:
    """Split a routing function's return annotation into named nodes and Send targets."""
    annotation = get_type_hints(route, include_extras=True)["return"]
    named: set[str] = set()
    sent: set[str] = set()
    for member in _union_members(annotation):
        if get_origin(member) is Annotated:
            base, *metadata = get_args(member)
            assert base == list[Send]
            for item in metadata:
                sent |= _literal_names(item)
        else:
            named |= _literal_names(member)
    return named, sent


def test_node_names_match_the_plan() -> None:
    assert graph.NODE_NAMES == PLAN_NODE_NAMES
    assert len(set(graph.NODE_NAMES)) == len(graph.NODE_NAMES) >= 5
    assert set(NODE_DEPENDENCIES) == set(graph.NODE_NAMES)


@pytest.mark.parametrize("name", PLAN_NODE_NAMES)
def test_every_node_is_a_traced_function_named_after_its_node(name: str) -> None:
    node = getattr(nodes, name)
    assert name in nodes.__all__
    assert not inspect.iscoroutinefunction(node), "nodes are sync (the UI streams with stream)"
    assert node.__name__ == name
    assert inspect.unwrap(node) is not node, "decorate the node with agentic_rag.tracing.traced"


@pytest.mark.parametrize("name", PLAN_NODE_NAMES)
def test_nodes_take_their_state_and_explicit_dependencies(name: str) -> None:
    node = getattr(nodes, name)
    hints = get_type_hints(node)
    state_param, *dependency_params = inspect.signature(node).parameters.values()
    assert hints[state_param.name] is (SubtaskInput if name in SEND_WORKERS else AgentState)
    assert hints["return"] == dict[str, Any]
    assert {param.name for param in dependency_params} == NODE_DEPENDENCIES[name]
    for param in dependency_params:
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default is inspect.Parameter.empty, "dependencies are bound, not defaulted"
        assert hints[param.name] == DEPENDENCY_TYPES[param.name]


def test_send_targets_are_exactly_the_nodes_that_take_a_subtask_input() -> None:
    workers = {
        name
        for name in graph.NODE_NAMES
        if next(iter(get_type_hints(getattr(nodes, name)).values())) is SubtaskInput
    }
    assert workers == set(get_args(routing.SubtaskWorker)) == SEND_WORKERS


@pytest.mark.parametrize("name", list(PLANNED_ROUTES))
def test_routing_annotations_list_the_planned_destinations(name: str) -> None:
    assert _route_targets(getattr(routing, name)) == PLANNED_ROUTES[name]


@pytest.mark.parametrize(
    "factory", [graph.build_agent_graph, tools.get_tools, tools.get_non_retrieval_tools]
)
def test_factories_require_their_settings(factory: Callable[..., object]) -> None:
    settings_param = next(iter(inspect.signature(factory).parameters.values()))
    assert settings_param.name == "settings"
    assert settings_param.default is inspect.Parameter.empty
    assert get_type_hints(factory)["settings"] is Settings


# --- routing --------------------------------------------------------------------------------


def test_route_after_analyze_follows_the_intent() -> None:
    retrieve = Subtask(id="s1", kind="retrieve", input="useState")
    tool = Subtask(id="s1", kind="tool", input="contrast", tool_name=CHECK_CONTRAST)

    assert (
        routing.route_after_analyze(AgentState(messages=[], intent="direct")) == "finalize_response"
    )
    assert routing.route_after_analyze(AgentState(messages=[], intent="complex")) == "plan_subtasks"
    (single,) = routing.route_after_analyze(
        AgentState(messages=[], intent="single", question="q", subtasks=[retrieve])
    )
    (call,) = routing.route_after_analyze(
        AgentState(messages=[], intent="tool", question="q", subtasks=[tool])
    )
    assert (single.node, single.arg) == ("run_rag_subtask", {"subtask": retrieve, "question": "q"})
    assert call.node == "call_tool"
    # Without a one-step plan the planner decides.
    assert routing.route_after_analyze(AgentState(messages=[], intent="single")) == "plan_subtasks"


def test_dispatch_sends_every_subtask_to_its_worker_in_plan_order() -> None:
    plan = [
        Subtask(id="s1", kind="retrieve", input="a"),
        Subtask(id="s2", kind="tool", input="b", tool_name=CHECK_CONTRAST),
        Subtask(id="s3", kind="retrieve", input="c"),
    ]

    sends = routing.dispatch_subtasks(AgentState(messages=[], question="q", subtasks=plan))

    assert [(send.node, send.arg["subtask"].id) for send in sends] == [
        ("run_rag_subtask", "s1"),
        ("call_tool", "s2"),
        ("run_rag_subtask", "s3"),
    ]


@pytest.mark.parametrize(
    ("verdict", "retry_count", "max_retries", "destination"),
    [
        ("grounded", 0, 2, "finalize_response"),
        ("insufficient", 0, 2, "plan_subtasks"),
        ("insufficient", 1, 2, "plan_subtasks"),
        ("insufficient", 2, 2, "finalize_response"),
        ("insufficient", 0, 0, "finalize_response"),
    ],
)
def test_route_after_verify_bounds_the_loop(
    verdict: str, retry_count: int, max_retries: int, destination: str
) -> None:
    state = AgentState(messages=[], verdict=verdict, retry_count=retry_count)  # type: ignore[typeddict-item]

    assert routing.route_after_verify(state, max_retries=max_retries) == destination


# --- analyze_request ------------------------------------------------------------------------


def test_analyze_request_writes_a_direct_reply() -> None:
    model = scripted((".", '{"intent": "direct", "question": "Hi?", "reply": "Hello!"}'))

    update = nodes.analyze_request(user("Hi!"), chat_model=model, tools=TOOLS)

    assert own_update(update) == {"question": "Hi?", "intent": "direct", "draft_answer": "Hello!"}
    assert update["trace"][0].summary == "Intent direct"


def test_analyze_request_plans_one_search_for_a_single_question() -> None:
    model = scripted((".", '{"intent": "single", "search_query": "React useState hook"}'))

    update = nodes.analyze_request(user("Mi az a useState?"), chat_model=model, tools=TOOLS)

    assert update["question"] == "Mi az a useState?", "the latest message when no question is given"
    assert update["subtasks"] == [Subtask(id="s1", kind="retrieve", input="React useState hook")]


def test_analyze_request_plans_one_tool_call() -> None:
    reply = (
        '{"intent": "tool", "question": "q", "tool_name": "check_contrast", '
        '"tool_args": {"foreground": "#777", "background": "#fff"}}'
    )

    update = nodes.analyze_request(user("q"), chat_model=scripted((".", reply)), tools=TOOLS)

    (subtask,) = update["subtasks"]
    assert (subtask.kind, subtask.tool_name, subtask.tool_args) == (
        "tool",
        "check_contrast",
        {"foreground": "#777", "background": "#fff"},
    )


@pytest.mark.parametrize(
    ("reply", "intent"),
    [
        ('{"intent": "direct"}', "single"),  # a direct intent without a reply
        ('{"intent": "tool", "tool_name": "calculator"}', "complex"),  # an unknown tool
        ("not JSON", "single"),  # an unreadable reply
    ],
)
def test_analyze_request_falls_back_on_unusable_replies(reply: str, intent: str) -> None:
    update = nodes.analyze_request(user("useState"), chat_model=scripted((".", reply)), tools=TOOLS)

    assert update["intent"] == intent
    if intent == "single":
        assert update["subtasks"] == [Subtask(id="s1", kind="retrieve", input="useState")]


def test_analyze_request_shows_the_history_and_the_tool_catalog() -> None:
    model = scripted((".", '{"intent": "complex"}'))
    state = AgentState(
        messages=[
            HumanMessage("What is useState?"),
            AIMessage("A Hook."),
            HumanMessage("And in Nuxt?"),
        ]
    )

    nodes.analyze_request(state, chat_model=model, tools=TOOLS)

    prompt = model.last_prompt or ""
    assert "User: What is useState?\nAssistant: A Hook.\n\nLatest message: And in Nuxt?" in prompt
    assert "- check_contrast: " in prompt and "- css_specificity: " in prompt


# --- plan_subtasks --------------------------------------------------------------------------


def test_plan_subtasks_numbers_the_plan_and_resets_the_results() -> None:
    reply = (
        '{"subtasks": [{"kind": "retrieve", "input": "Nuxt useState"}, '
        '{"kind": "retrieve", "input": "React useState"}, '
        '{"kind": "tool", "tool_name": "check_contrast", '
        '"tool_args": {"foreground": "#000", "background": "#fff"}}, '
        '{"kind": "tool", "tool_name": "calculator", "input": "2+2"}, '
        '{"kind": "retrieve", "input": "react USESTATE"}]}'
    )
    state = AgentState(messages=[HumanMessage("q")], question="q")

    update = nodes.plan_subtasks(state, chat_model=scripted((".", reply)), tools=TOOLS)

    plan = update["subtasks"]
    assert [(s.id, s.kind, s.tool_name or s.input) for s in plan] == [
        ("s1", "retrieve", "Nuxt useState"),
        ("s2", "retrieve", "React useState"),
        ("s3", "tool", "check_contrast"),
        ("s4", "retrieve", "2+2"),  # an unknown tool becomes a search
    ]  # the repeated search is dropped
    assert isinstance(update["subtask_results"], Overwrite)
    assert update["subtask_results"].value == []


def test_the_plan_is_capped_and_never_empty() -> None:
    many = ", ".join(f'{{"kind": "retrieve", "input": "q{n}"}}' for n in range(9))
    state = AgentState(messages=[HumanMessage("q")], question="the question")

    capped = nodes.plan_subtasks(
        state, chat_model=scripted((".", f'{{"subtasks": [{many}]}}')), tools={}
    )
    empty = nodes.plan_subtasks(state, chat_model=scripted((".", '{"subtasks": []}')), tools={})
    unreadable = nodes.plan_subtasks(state, chat_model=scripted((".", "a plan")), tools={})

    assert len(capped["subtasks"]) == MAX_SUBTASKS
    assert (
        empty["subtasks"]
        == unreadable["subtasks"]
        == [Subtask(id="s1", kind="retrieve", input="the question")]
    )


def test_a_replan_keeps_the_good_results_and_plans_only_what_is_missing() -> None:
    previous = [
        Subtask(id="s1", kind="retrieve", input="Nuxt useState"),
        Subtask(id="s2", kind="retrieve", input="React useState"),
    ]
    kept = retrieved("s1", source("c1"))
    failed = SubtaskResult(subtask_id="s2", kind="retrieve", output="", ok=False, error="boom")
    state = AgentState(
        messages=[HumanMessage("q")],
        question="q",
        subtasks=previous,
        subtask_results=[kept, failed],
        verdict="insufficient",
        critique="React is not covered",
    )
    reply = (
        '{"subtasks": [{"kind": "retrieve", "input": "Nuxt useState"}, '
        '{"kind": "retrieve", "input": "React useState hook"}]}'
    )
    model = scripted((".", reply))

    update = nodes.plan_subtasks(state, chat_model=model, tools={})

    assert update["subtask_results"].value == [kept]
    assert update["subtasks"] == [Subtask(id="s2", kind="retrieve", input="React useState hook")]
    prompt = model.last_prompt or ""
    assert "because: React is not covered" in prompt
    assert "Already available, do not plan again:\n- retrieve: Nuxt useState" in prompt


# --- the Send workers -----------------------------------------------------------------------


def test_run_rag_subtask_returns_the_sources_and_forwards_the_trace() -> None:
    event = TraceEvent(node="retrieve", started_at=1.0, ended_at=1.5, duration_ms=500.0)

    def rag(rag_input: RagInput) -> RagOutput:
        return RagOutput(context=f"[1] {rag_input['query']}", sources=[source("c1")], trace=[event])

    search = SearchKnowledgeBaseTool(rag_graph=RunnableLambda(rag))
    subtask = Subtask(id="s1", kind="retrieve", input="useState")

    update = nodes.run_rag_subtask(SubtaskInput(subtask=subtask, question="q"), search_tool=search)

    assert update["subtask_results"] == [
        SubtaskResult(
            subtask_id="s1", kind="retrieve", output="[1] useState", sources=[source("c1")]
        )
    ]
    assert [e.node for e in update["trace"]] == ["retrieve", "run_rag_subtask"]
    assert update["trace"][-1].summary == "1 source found"


def test_a_tool_exception_of_the_search_becomes_a_failed_result() -> None:
    def broken(rag_input: RagInput) -> RagOutput:
        raise ToolException("the search failed")

    search = SearchKnowledgeBaseTool(rag_graph=RunnableLambda(broken))
    subtask = Subtask(id="s1", kind="retrieve", input="q")

    update = nodes.run_rag_subtask(SubtaskInput(subtask=subtask, question="q"), search_tool=search)

    (result,) = update["subtask_results"]
    assert (result.ok, result.error) == (False, "the search failed")


def tool_subtask(name: str | None, args: dict[str, Any]) -> SubtaskInput:
    """The Send payload of one tool sub-task."""
    subtask = Subtask(
        id="s1", kind="tool", input="compute", tool_name=name or "missing", tool_args=args
    )
    return SubtaskInput(subtask=subtask, question="q")


@pytest.mark.parametrize(
    ("name", "args", "ok", "text"),
    [
        (
            CHECK_CONTRAST,
            {"foreground": "#777", "background": "#fff"},
            True,
            "Contrast ratio 4.47:1",
        ),
        ("calculator", {}, False, "unknown tool 'calculator'"),
        (
            CHECK_CONTRAST,
            {"foreground": "#777"},
            False,
            "invalid arguments for check_contrast: background",
        ),
        (CHECK_CONTRAST, {"foreground": "#12", "background": "#fff"}, False, "unsupported colour"),
    ],
)
def test_call_tool_runs_the_tool_or_reports_why_not(
    name: str, args: dict[str, Any], ok: bool, text: str
) -> None:
    update = nodes.call_tool(tool_subtask(name, args), tools=TOOLS)

    (result,) = update["subtask_results"]
    assert result.ok is ok
    assert text in (result.output if ok else result.error or "")


# --- synthesis, verification and the final answer -------------------------------------------


def test_sources_are_numbered_globally_without_repeats() -> None:
    results = [
        retrieved("s1", source("a"), source("b")),
        SubtaskResult(subtask_id="s2", kind="tool", output="4.47:1"),
        retrieved("s3", source("b"), source("c")),
    ]

    assert [s.chunk_id for s in numbered_sources(results)] == ["a", "b", "c"]


def test_synthesize_answer_cites_from_the_global_numbering() -> None:
    results = [
        retrieved("s1", source("a", "Nuxt state.")),
        retrieved("s2", source("a", "Nuxt state."), source("b", "React state.")),
        SubtaskResult(subtask_id="s3", kind="tool", output="Contrast ratio 4.47:1."),
        SubtaskResult(subtask_id="s4", kind="tool", output="", ok=False, error="unknown tool"),
    ]
    model = scripted((".", "<think>plan</think>Nuxt [1], React [2]."))
    state = AgentState(
        messages=[HumanMessage("Hasonlítsd össze")], question="Compare", subtask_results=results
    )

    update = nodes.synthesize_answer(state, chat_model=model)

    assert update["draft_answer"] == "Nuxt [1], React [2]."
    prompt = model.last_prompt or ""
    assert prompt.startswith(SYNTHESIZE_INSTRUCTIONS)
    assert (
        "User's message: Hasonlítsd össze\nQuestion: Compare\n\nDocumentation excerpts:\n"
        "[1] Title\n\nNuxt state.\n\n[2] Title\n\nReact state." in prompt
    )
    assert "Tool results:\n- Contrast ratio 4.47:1." in prompt
    assert "Failed steps:\n- tool s4: unknown tool" in prompt


def test_an_empty_draft_is_replaced() -> None:
    state = AgentState(messages=[HumanMessage("q")], subtask_results=[])

    update = nodes.synthesize_answer(state, chat_model=scripted((".", "<think>...</think>")))

    assert update["draft_answer"] == NO_DRAFT


def test_material_without_excerpts_says_so_only_without_tool_results() -> None:
    tool_result = SubtaskResult(subtask_id="s1", kind="tool", output="4.47:1")

    assert format_material([], []) == "Documentation excerpts: none found."
    assert format_material([], [tool_result]) == "Tool results:\n- 4.47:1"


@pytest.mark.parametrize(
    ("reply", "previous", "verdict", "critique", "retry_count"),
    [
        ('{"grounded": true}', None, "grounded", "", 0),
        ('{"grounded": false, "missing": "React"}', None, "insufficient", "React", 0),
        ('{"grounded": false, "missing": "React"}', "insufficient", "insufficient", "React", 2),
        (
            "unreadable",
            "insufficient",
            "unavailable",
            "Verification returned an unreadable response. Please try again.",
            2,
        ),
    ],
)
def test_verify_answer_judges_the_draft_and_counts_the_replans(
    reply: str, previous: str | None, verdict: str, critique: str, retry_count: int
) -> None:
    state = AgentState(
        messages=[HumanMessage("q")],
        question="q",
        draft_answer="draft",
        subtask_results=[retrieved("s1", source("a"))],
    )
    if previous:
        state["verdict"] = previous  # type: ignore[typeddict-item]
        state["retry_count"] = 1
    model = scripted((".", reply))

    update = nodes.verify_answer(state, chat_model=model)

    assert own_update(update) == {
        "verdict": verdict,
        "critique": critique,
        "retry_count": retry_count,
    }
    assert (model.last_prompt or "").startswith(VERIFY_INSTRUCTIONS)


def test_finalize_response_returns_the_numbered_sources_and_the_message() -> None:
    state = AgentState(
        messages=[HumanMessage("q")],
        intent="single",
        draft_answer="Use it [1], see also [3].",
        subtask_results=[retrieved("s1", source("a"), source("b"))],
        verdict="grounded",
    )

    update = nodes.finalize_response(state)

    assert update["answer"] == "Use it [1], see also."
    assert [s.chunk_id for s in update["sources"]] == ["a", "b"]
    assert update["messages"] == [AIMessage(content="Use it [1], see also.")]


def test_finalize_response_shows_tool_output_verbatim() -> None:
    contrast = Subtask(id="s1", kind="tool", input="q", tool_name=CHECK_CONTRAST)
    state = AgentState(
        messages=[HumanMessage("q")],
        intent="tool",
        draft_answer="Megfelel.",
        subtasks=[contrast],
        subtask_results=[
            SubtaskResult(
                subtask_id="s1", kind="tool", output="Contrast ratio 4.47:1.\n- AA: fails"
            ),
        ],
        verdict="grounded",
    )

    answer = nodes.finalize_response(state)["answer"]

    assert answer == (
        "Megfelel.\n\n**Tool result** (`check_contrast`):\n\n"
        "```text\nContrast ratio 4.47:1.\n- AA: fails\n```"
    )


def test_finalize_response_names_kept_tool_results_and_shows_a_repeated_check_once() -> None:
    # s1 ran in a rejected round and was kept; the re-plan ran the same check again as s2.
    support = Subtask(id="s2", kind="tool", input="q", tool_name="browser_support")
    output = "Safari 15: not supported (added in 15.4)."
    kept = SubtaskResult(subtask_id="s1", kind="tool", output=output, tool_name="browser_support")
    again = SubtaskResult(subtask_id="s2", kind="tool", output=output, tool_name="browser_support")
    state = AgentState(
        messages=[HumanMessage("q")],
        intent="tool",
        draft_answer="No.",
        subtasks=[support],
        subtask_results=[kept, again],
        verdict="grounded",
    )

    answer = nodes.finalize_response(state)["answer"]

    assert answer == f"No.\n\n**Tool result** (`browser_support`):\n\n```text\n{output}\n```"


def test_finalize_response_marks_a_partial_answer_and_keeps_direct_replies_clean() -> None:
    partial = AgentState(
        messages=[HumanMessage("q")],
        intent="complex",
        draft_answer="Half [1].",
        subtask_results=[retrieved("s1", source("a"))],
        verdict="insufficient",
        critique="React is not covered",
    )
    direct = AgentState(messages=[HumanMessage("hi")], intent="direct", draft_answer="Hello [1]!")

    assert nodes.finalize_response(partial)["answer"] == (
        f"Half [1].\n\n*{PARTIAL_NOTE}: React is not covered*"
    )
    assert own_update(nodes.finalize_response(direct))["sources"] == []
    assert nodes.finalize_response(direct)["answer"] == "Hello!"


# --- the compiled graph ---------------------------------------------------------------------


@pytest.fixture
def indexed(settings: Settings) -> Settings:
    """Fake settings with a three-page corpus and its index."""
    pages = {
        "nuxt.md": "---\ntitle: useState\n---\n\nThe Nuxt useState composable shares state.\n",
        "react.md": "---\ntitle: useState\n---\n\nThe React useState Hook adds a state variable.\n",
        "has.md": "---\ntitle: :has()\n---\n\nThe :has() pseudo-class selects a parent element.\n",
    }
    for name, text in pages.items():
        (settings.data_dir / name).write_text(text, encoding="utf-8")
    build_index(settings)
    return settings


def ask(settings: Settings, text: str) -> dict[str, Any]:
    """Run the compiled graph on one user message."""
    return graph.build_agent_graph(settings).invoke({"messages": [("user", text)]})


def main_steps(output: dict[str, Any]) -> list[str]:
    """The main graph's own steps of a run, in order."""
    return [event.node for event in output["trace"] if event.node in PLAN_NODE_NAMES]


def test_building_the_graph_loads_nothing(settings: Settings) -> None:
    compiled = graph.build_agent_graph(settings)

    assert set(compiled.get_graph().nodes) >= set(PLAN_NODE_NAMES)


def test_a_greeting_is_answered_directly(settings: Settings) -> None:
    output = ask(settings, "Hello!")

    assert output["intent"] == "direct"
    assert main_steps(output) == ["analyze_request", "finalize_response"]
    assert output["answer"].startswith("Hi! I answer questions about web frontend development")
    assert output["sources"] == []
    assert output["messages"][-1].content == output["answer"]


def test_a_single_question_searches_once_and_cites_the_sources(indexed: Settings) -> None:
    output = ask(indexed, "How does the React useState Hook work?")

    assert output["intent"] == "single"
    assert main_steps(output) == [
        "analyze_request",
        "run_rag_subtask",
        "synthesize_answer",
        "verify_answer",
        "finalize_response",
    ]
    assert output["sources"][0].source == "react.md"
    assert "[1]" in output["answer"]
    # The RAG subgraph's events are forwarded into the main trace.
    assert {"rewrite_query", "retrieve", "grade_documents", "build_context"} <= {
        event.node for event in output["trace"]
    }


def test_a_comparison_is_split_into_parallel_searches(indexed: Settings) -> None:
    output = ask(indexed, "Nuxt useState and React useState")

    assert output["intent"] == "complex"
    assert main_steps(output).count("run_rag_subtask") == 2
    assert [subtask.input for subtask in output["subtasks"]] == ["Nuxt useState", "React useState"]
    assert {result.subtask_id for result in output["subtask_results"]} == {"s1", "s2"}
    chunk_ids = [s.chunk_id for s in output["sources"]]
    assert len(chunk_ids) == len(set(chunk_ids)), "sources are de-duplicated across sub-tasks"


def test_a_tool_question_runs_the_tool(settings: Settings) -> None:
    output = ask(settings, "Does #777777 text on white pass WCAG AA?")

    assert output["intent"] == "tool"
    assert main_steps(output) == [
        "analyze_request",
        "call_tool",
        "synthesize_answer",
        "finalize_response",
    ]
    assert "Contrast ratio 4.47:1 for #777777 on #ffffff" in output["answer"]


def test_the_replan_loop_is_bounded(indexed: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    never_satisfied = scripted(
        (r"(?s)\AYou are the router", '{"intent": "single"}'),
        (r"(?s)\AYou split a request", '{"subtasks": [{"kind": "retrieve", "input": "more"}]}'),
        (r"(?s)\AYou check a draft", '{"grounded": false, "missing": "everything"}'),
        (r".", "A draft [1]."),
    )
    monkeypatch.setattr(graph, "get_chat_model", lambda settings: never_satisfied)

    output = ask(indexed, "useState")

    assert main_steps(output).count("plan_subtasks") == indexed.max_retries == 2
    assert main_steps(output).count("verify_answer") == 3
    assert output["verdict"] == "insufficient"
    assert output["retry_count"] == 2
    assert output["answer"].endswith(f"*{PARTIAL_NOTE}: everything*")


def test_export_graph_draws_the_main_workflow(
    settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["export-graph", "--graph", "agent", "--format", "mermaid"]) == 0

    out = capsys.readouterr().out
    for edge in (
        "__start__ --> analyze_request;",
        "analyze_request -.-> plan_subtasks;",
        "plan_subtasks -.-> run_rag_subtask;",
        "run_rag_subtask --> synthesize_answer;",
        "verify_answer -.-> plan_subtasks;",
        "finalize_response --> __end__;",
    ):
        assert edge in out


# --- the fake rules and the prompts ---------------------------------------------------------


def first_rule(messages: list[Any]) -> FakeRule:
    """The default fake rule that answers a prompt."""
    prompt = render_prompt(messages)
    return next(rule for rule in DEFAULT_FAKE_RULES if rule.matches(prompt))


@pytest.mark.parametrize(
    ("question", "intent"),
    [
        ("Hello!", '"direct"'),
        ("Does #777 text on white pass AA?", '"check_contrast"'),
        ("Is `:has()` supported in Safari 15?", '"browser_support"'),
        ("Which specificity is higher: `#a` or `.b .c`?", '"css_specificity"'),
        ("Nuxt useState vs React useState", '"complex"'),
        ("How does useEffect work?", '"single"'),
    ],
)
def test_the_fake_rules_recognise_the_analysis_prompt(question: str, intent: str) -> None:
    rule = first_rule(analyze_messages([], question, TOOLS))

    assert intent in rule.reply


def test_the_fake_rules_recognise_the_other_prompts() -> None:
    material = format_material([source("a")], [])

    assert '"subtasks"' in first_rule(plan_messages("a and b", TOOLS, max_subtasks=5)).reply
    assert "source [1]" in first_rule(synthesize_messages("q", material)).reply
    assert '"grounded": true' in first_rule(verify_messages("q", material, "draft")).reply


def test_the_prompts_start_with_the_sentences_the_fake_rules_expect() -> None:
    assert ANALYZE_INSTRUCTIONS.startswith("You are the router of a frontend developer assistant.")
    assert PLAN_INSTRUCTIONS.startswith("You split a request")
    assert SYNTHESIZE_INSTRUCTIONS.startswith("You are a frontend developer assistant. Answer")
    assert VERIFY_INSTRUCTIONS.startswith("You check a draft answer")
