"""Tests of the Streamlit UI: the entrypoint under AppTest and the components on their own.

The app runs headless and offline with ``streamlit.testing.v1.AppTest`` in fake mode. The main
graph module ``agentic_rag.agent.graph`` is replaced in ``sys.modules`` by stand-ins, so the
tests do not depend on the real module (a stub until Phase 4): a builder that raises
NotImplementedError like the stub, one that raises an unexpected error, and builders that
return small real LangGraph graphs on the project's state contracts.
"""

import functools
import importlib.abc
import logging
import sys
import types
from collections.abc import Callable, Iterator, Sequence
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any

import pytest
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send
from streamlit.testing.v1 import AppTest

import agentic_rag.ui
from agentic_rag.agent.state import (
    AgentInput,
    AgentOutput,
    AgentState,
    Subtask,
    SubtaskInput,
    SubtaskResult,
)
from agentic_rag.config import Settings
from agentic_rag.rag.state import Source
from agentic_rag.tracing import TraceEvent, epoch_now, traced
from agentic_rag.ui.components import (
    ChatTurn,
    escape_markdown,
    format_duration,
    group_parallel_steps,
    settings_summary,
)

APP_PATH = Path(agentic_rag.ui.__file__).with_name("app.py")
GRAPH_MODULE = "agentic_rag.agent.graph"
HISTORY_KEY = "chat_history"
PHASE_4_MESSAGE = (
    "agentic_rag.agent.graph.build_agent_graph is planned for Phase 4 "
    "(see docs/project-structure-plan.md, section 8)"
)
NODE_STUB_MESSAGE = (
    "agentic_rag.agent.nodes.plan_subtasks is planned for Phase 4 "
    "(see docs/project-structure-plan.md, section 8)"
)
HEAVY_MODULES = frozenset(
    {
        "chromadb",
        "langchain_chroma",
        "langchain_huggingface",
        "sentence_transformers",
        "torch",
        "transformers",
    }
)
TIMEOUT = 30.0
"""Seconds an AppTest run may take: generous for a cold start; a run takes well under one."""

T0 = 1_759_300_000.0
"""Start of the hand-made trace events, in epoch seconds."""

DEMO_SOURCE = Source(
    chunk_id="handbook-12-3",
    source="handbook.pdf",
    content="Employees get 25 days of paid leave per year.",
    title="Employee handbook",
    page=12,
    section="Leave",
    score=0.875,
)
MAIN_STEPS = [
    "analyze_request",
    "plan_subtasks",
    "run_rag_subtask",
    "call_tool",
    "finalize_response",
]


# ---------------------------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_streamlit_globals() -> Iterator[None]:
    """Clear Streamlit's process-wide resource cache and restore the root log level.

    ``st.cache_resource`` outlives an AppTest run, so a graph cached in one test would answer
    in the next one; the app's ``configure_logging()`` call changes the root logger level.
    """
    root = logging.getLogger()
    level = root.level
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()
    root.setLevel(level)


class _HeavyImportBlocker(importlib.abc.MetaPathFinder):
    """Meta path finder that refuses the heavy libraries and records every attempt."""

    def __init__(self) -> None:
        self.attempts: list[str] = []

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: types.ModuleType | None = None,
    ) -> ModuleSpec | None:
        """Refuse a heavy module; leave every other module to the next finders."""
        if fullname.partition(".")[0] in HEAVY_MODULES:
            self.attempts.append(fullname)
            msg = f"{fullname} must not be imported when the app starts"
            raise ModuleNotFoundError(msg, name=fullname)
        return None


@pytest.fixture
def heavy_imports(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Hide and block the heavy libraries during one test.

    Heavy modules that an earlier test imported are taken out of ``sys.modules`` for the test
    (monkeypatch puts them back), so an import by the app cannot hide behind them.

    Returns:
        The names of the heavy modules something tried to import during the test.
    """
    for name in list(sys.modules):
        if name.partition(".")[0] in HEAVY_MODULES:
            monkeypatch.delitem(sys.modules, name)
    blocker = _HeavyImportBlocker()
    monkeypatch.setattr(sys, "meta_path", [blocker, *sys.meta_path])
    return blocker.attempts


def install_graph_module(
    monkeypatch: pytest.MonkeyPatch, build: Callable[[Settings | None], Any]
) -> list[Settings | None]:
    """Replace ``agentic_rag.agent.graph`` with a stand-in whose builder delegates to build.

    Returns:
        The settings of every ``build_agent_graph`` call, in call order.
    """
    calls: list[Settings | None] = []

    def build_agent_graph(settings: Settings | None = None) -> Any:
        calls.append(settings)
        return build(settings)

    module = types.ModuleType(GRAPH_MODULE)
    module.build_agent_graph = build_agent_graph
    monkeypatch.setitem(sys.modules, GRAPH_MODULE, module)
    return calls


@functools.cache
def _scanned_component_manager() -> Any:
    """Return the custom-component registry of one AppTest run; scanned once per session."""
    at = AppTest.from_string("import streamlit", default_timeout=TIMEOUT).run()
    return getattr(at, "_bidi_component_manager", None)


def share_component_scan(at: AppTest) -> AppTest:
    """Let a new AppTest reuse the custom-component registry an earlier one scanned.

    On its first run, every AppTest scans the metadata of all installed distributions for
    custom components (about 0.2 s here); the app uses none, so one scan per session keeps
    this file fast. This relies on a private attribute: if Streamlit renames it, every
    AppTest scans again, which is slower but still correct.
    """
    manager = _scanned_component_manager()
    if manager is not None and getattr(at, "_bidi_component_manager", ...) is None:
        at._bidi_component_manager = manager
    return at


def start_app() -> AppTest:
    """Run the entrypoint once, as a browser that opens the page does."""
    return share_component_scan(AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)).run()


def run_script(script: Callable[..., None], *args: Any) -> AppTest:
    """Run a component script with ``AppTest.from_function``, passing it ``args``."""
    at = AppTest.from_function(script, args=args, default_timeout=TIMEOUT)
    return share_component_scan(at).run()


def ask(at: AppTest, question: str) -> AppTest:
    """Submit a question through the chat input and run the app."""
    return at.chat_input[0].set_value(question).run()


def step_names(block: Any) -> list[str]:
    """Return the node names a step panel shows, in display order.

    ``block.status[0]`` is the panel itself; the other entries are its timeline steps. A
    group of parallel workers lists its node names in a table.
    """
    names: list[str] = []
    for step in block.status[1:]:
        if " in parallel" in step.label:
            names.extend(cell.strip("`") for cell in step.table[0].value["Step"])
        else:
            names.append(step.label.split(" · ")[0].strip("`"))
    return names


def make_event(
    node: str, start_ms: float, end_ms: float, summary: str = "", **metadata: Any
) -> TraceEvent:
    """Return a trace event that ran from ``start_ms`` to ``end_ms`` after ``T0``."""
    return TraceEvent(
        node=node,
        started_at=T0 + start_ms / 1000,
        ended_at=T0 + end_ms / 1000,
        duration_ms=end_ms - start_ms,
        summary=summary,
        metadata=metadata,
    )


# Scripts for AppTest.from_function: the body runs as a page, so it imports what it uses.


def trace_panel_script(events: list, state: str) -> None:
    """Render one step panel."""
    from agentic_rag.ui.components import render_trace

    render_trace(events, state=state)


def sources_panel_script(sources: list) -> None:
    """Render one retrieved-context panel."""
    from agentic_rag.ui.components import render_sources

    render_sources(sources)


# ---------------------------------------------------------------------------------------------
# Stand-in graph builders
# ---------------------------------------------------------------------------------------------


def raise_not_implemented(settings: Settings | None) -> Any:
    """Behave like the Phase 4 stub of ``build_agent_graph``."""
    raise NotImplementedError(PHASE_4_MESSAGE)


def raise_runtime_error(settings: Settings | None) -> Any:
    """Fail the way an unexpected bug would."""
    raise RuntimeError("the index exploded")


@traced(summarize=lambda update: f"intent: {update['intent']}")
def analyze_request(state: AgentState) -> dict[str, Any]:
    """Read the question from the last message and choose the complex route."""
    return {"question": str(state["messages"][-1].content), "intent": "complex"}


@traced(summarize=lambda update: f"{len(update['subtasks'])} sub-tasks")
def plan_subtasks(state: AgentState) -> dict[str, Any]:
    """Plan one retrieval and one tool call."""
    return {
        "subtasks": [
            Subtask(id="s1", kind="retrieve", input=state["question"]),
            Subtask(id="s2", kind="tool", input="compute", tool_name="calculator"),
        ]
    }


@traced
def plan_subtasks_stub(state: AgentState) -> dict[str, Any]:
    """Behave like a node that is still a stub."""
    raise NotImplementedError(NODE_STUB_MESSAGE)


def dispatch_subtasks(state: AgentState) -> list[Send]:
    """Send every sub-task to the worker node of its kind."""
    return [
        Send(
            "run_rag_subtask" if subtask.kind == "retrieve" else "call_tool",
            {"subtask": subtask, "question": state["question"]},
        )
        for subtask in state["subtasks"]
    ]


@traced(summarize=lambda update: "1 chunk")
def run_rag_subtask(payload: SubtaskInput) -> dict[str, Any]:
    """Return a canned retrieval result and forward a RAG subgraph event, like the worker."""
    now = epoch_now()
    forwarded = TraceEvent(node="retrieve", started_at=now, ended_at=now, duration_ms=0.0)
    result = SubtaskResult(
        subtask_id=payload["subtask"].id,
        kind="retrieve",
        output=DEMO_SOURCE.content,
        sources=[DEMO_SOURCE],
    )
    return {"subtask_results": [result], "trace": [forwarded]}


@traced(summarize=lambda update: "42")
def call_tool(payload: SubtaskInput) -> dict[str, Any]:
    """Return a canned tool result."""
    result = SubtaskResult(subtask_id=payload["subtask"].id, kind="tool", output="42")
    return {"subtask_results": [result]}


@traced
def finalize_response(state: AgentState) -> dict[str, Any]:
    """Answer with the number of user messages seen, the question and a citation."""
    user_messages = sum(isinstance(message, HumanMessage) for message in state["messages"])
    answer = f"Answer {user_messages} to *{state['question']}* [1]"
    sources = [source for result in state["subtask_results"] for source in result.sources]
    return {"answer": answer, "sources": sources, "messages": [AIMessage(content=answer)]}


def build_demo_graph(settings: Settings | None = None) -> CompiledStateGraph:
    """Build a small real graph on the state contracts, standing in for the Phase 4 graph.

    ``analyze_request -> plan_subtasks -> Send(run_rag_subtask | call_tool) ->
    finalize_response``; ``run_rag_subtask`` forwards a RAG subgraph event.
    """
    builder = StateGraph(AgentState, input_schema=AgentInput, output_schema=AgentOutput)
    for node in (analyze_request, plan_subtasks, run_rag_subtask, call_tool, finalize_response):
        builder.add_node(node.__name__, node)
    builder.add_edge(START, "analyze_request")
    builder.add_edge("analyze_request", "plan_subtasks")
    builder.add_conditional_edges(
        "plan_subtasks", dispatch_subtasks, ["run_rag_subtask", "call_tool"]
    )
    builder.add_edge("run_rag_subtask", "finalize_response")
    builder.add_edge("call_tool", "finalize_response")
    builder.add_edge("finalize_response", END)
    return builder.compile()


def build_partial_graph(settings: Settings | None = None) -> CompiledStateGraph:
    """Build a graph whose second node is still a stub raising NotImplementedError."""
    builder = StateGraph(AgentState, input_schema=AgentInput, output_schema=AgentOutput)
    builder.add_node("analyze_request", analyze_request)
    builder.add_node("plan_subtasks", plan_subtasks_stub)
    builder.add_edge(START, "analyze_request")
    builder.add_edge("analyze_request", "plan_subtasks")
    builder.add_edge("plan_subtasks", END)
    return builder.compile()


# ---------------------------------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------------------------------


def test_app_starts_offline_in_fake_mode_without_loading_heavy_libraries(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, heavy_imports: list[str]
) -> None:
    # Import both modules afresh during the run, so their imports go through the blocker.
    monkeypatch.delitem(sys.modules, GRAPH_MODULE, raising=False)
    monkeypatch.delitem(sys.modules, "agentic_rag.ui.components")

    at = start_app()

    assert not at.exception
    assert at.title[0].value == "Agentic RAG chatbot"
    assert at.main.caption[0].value.startswith("Prototype:")
    assert at.chat_input[0].placeholder == "Ask a question"
    assert not at.chat_message
    assert heavy_imports == []
    assert GRAPH_MODULE not in sys.modules


def test_sidebar_shows_the_effective_configuration_read_only(settings: Settings) -> None:
    at = start_app()

    table = at.sidebar.table[0].value
    assert dict(zip(table.index, table["value"], strict=True)) == settings_summary(settings)
    assert [button.label for button in at.sidebar.button] == ["Clear conversation"]
    for widgets in (
        at.sidebar.text_input,
        at.sidebar.number_input,
        at.sidebar.selectbox,
        at.sidebar.slider,
        at.sidebar.toggle,
        at.sidebar.checkbox,
    ):
        assert not widgets


def test_sidebar_shows_ollama_settings_without_contacting_the_server(
    monkeypatch: pytest.MonkeyPatch, heavy_imports: list[str]
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.1:8b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.invalid:11434")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "huggingface")
    monkeypatch.setenv("TOP_K", "6")

    at = start_app()

    assert not at.exception
    assert at.sidebar.table[0].value["value"].tolist() == [
        "`ollama` · `llama3.1:8b`",
        "`http://ollama.invalid:11434`",
        "`huggingface` · `intfloat/multilingual-e5-small`",
        "6",
    ]
    assert heavy_imports == []


def test_question_shows_the_not_implemented_notice_and_keeps_the_history(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = install_graph_module(monkeypatch, raise_not_implemented)
    at = start_app()

    ask(at, "What is the leave policy?")

    assert not at.exception
    user, assistant = at.chat_message
    assert (user.name, user.text[0].value) == ("user", "What is the leave policy?")
    assert assistant.name == "assistant"
    notice = assistant.info[0]
    assert notice.proto.title == "Not available yet"
    assert escape_markdown(PHASE_4_MESSAGE) in notice.value
    # The places of the steps and of the retrieved context are there, empty.
    assert [caption.value for caption in assistant.caption] == [
        ":material/account_tree: No agent steps were recorded for this turn.",
        ":material/description: No context was retrieved for this turn.",
    ]

    ask(at, "And the sick leave?")
    at.run()  # A rerun without a new question replays the history and runs nothing.

    assert not at.exception
    assert [message.name for message in at.chat_message] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert [message.text[0].value for message in at.chat_message[::2]] == [
        "What is the leave policy?",
        "And the sick leave?",
    ]
    assert all(message.info for message in at.chat_message[1::2])
    history = at.session_state[HISTORY_KEY]
    assert [turn.notice for turn in history] == [None, PHASE_4_MESSAGE, None, PHASE_4_MESSAGE]
    # One build attempt per question: a failed build is not cached.
    assert [call.llm_provider if call else None for call in calls] == ["fake", "fake"]


def test_not_implemented_inside_the_run_keeps_the_steps_before_it(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_graph_module(monkeypatch, build_partial_graph)
    at = start_app()

    ask(at, "What is the leave policy?")

    assert not at.exception
    assistant = at.chat_message[1]
    assert escape_markdown(NODE_STUB_MESSAGE) in assistant.info[0].value
    assert assistant.status[0].label.startswith("Agent steps · 1 step · ")
    assert step_names(assistant) == ["analyze_request"]


def test_unexpected_error_is_shown_with_st_exception_and_the_session_goes_on(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_graph_module(monkeypatch, raise_runtime_error)
    at = start_app()

    ask(at, "Will this work?")

    assert [error.value for error in at.chat_message[1].exception] == ["the index exploded"]
    assert at.chat_input

    ask(at, "And now?")

    assert [message.name for message in at.chat_message] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert [error.value for error in at.exception] == ["the index exploded"] * 2
    assert all(isinstance(turn.error, RuntimeError) for turn in at.session_state[HISTORY_KEY][1::2])


def test_answer_shows_the_steps_the_reply_and_the_retrieved_context(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = install_graph_module(monkeypatch, build_demo_graph)
    at = start_app()

    ask(at, "How many leave days?")

    assert not at.exception
    assistant = at.chat_message[1]
    panel = assistant.status[0]
    assert panel.icon == ":material/account_tree:"
    assert panel.label.startswith("Agent steps · 5 steps · ")
    shown = step_names(assistant)
    # The RAG subgraph event that run_rag_subtask forwards is not a main step.
    assert sorted(shown) == sorted(MAIN_STEPS)
    assert (shown[:2], shown[-1]) == (MAIN_STEPS[:2], MAIN_STEPS[-1])
    assert "Answer 1 to *How many leave days?* [1]" in [md.value for md in assistant.markdown]
    (chunk,) = assistant.expander
    assert chunk.label == r"\[1\] Employee handbook · p. 12 · score 0.875"
    assert chunk.text[0].value == DEMO_SOURCE.content
    answer_turn = at.session_state[HISTORY_KEY][1]
    assert answer_turn.sources == [DEMO_SOURCE]
    assert sorted(event.node for event in answer_turn.trace) == sorted(MAIN_STEPS)

    ask(at, "And for part-time staff?")

    assert not at.exception
    # The graph saw the whole conversation: two user messages.
    assert "Answer 2 to *And for part-time staff?* [1]" in [
        md.value for md in at.chat_message[3].markdown
    ]
    assert len(calls) == 1  # Built once, then served from st.cache_resource.


def test_clear_conversation_starts_over(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_graph_module(monkeypatch, raise_not_implemented)
    at = start_app()
    ask(at, "First question")
    assert len(at.chat_message) == 2

    at.sidebar.button[0].click().run()

    assert not at.exception
    assert not at.chat_message
    assert at.session_state[HISTORY_KEY] == []


def test_invalid_settings_show_an_error_instead_of_the_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TOP_K", "0")

    at = start_app()

    assert not at.exception
    assert at.error[0].proto.title == "Invalid configuration"
    assert "`TOP_K`" in at.error[0].value
    assert not at.chat_input


# ---------------------------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------------------------


def test_render_trace_shows_steps_in_order_and_groups_parallel_workers() -> None:
    events = [
        make_event("finalize_response", 91, 92),
        make_event("run_rag_subtask", 21, 80, "s1: 2 chunks", chunks=2),
        make_event("analyze_request", 0, 10, "intent: complex"),
        make_event("call_tool", 22, 25, "s2: 42"),
        make_event("plan_subtasks", 11, 20, "2 sub-tasks"),
        make_event("run_rag_subtask", 21, 90, "s3: 1 chunk", chunks=1),
    ]

    at = run_script(trace_panel_script, events, "complete")

    assert not at.exception
    panel, *steps = at.status
    assert (panel.label, panel.icon, panel.proto.expanded) == (
        "Agent steps · 6 steps · 92 ms",
        ":material/account_tree:",
        False,
    )
    assert [step.label for step in steps] == [
        "`analyze_request` · 10 ms",
        "`plan_subtasks` · 9.0 ms",
        "3 steps in parallel · 69 ms",
        "`finalize_response` · 1.0 ms",
    ]
    assert steps[0].markdown[0].value == "intent: complex"
    assert steps[0].caption[0].value == "Started at +0 ms"
    assert steps[3].caption[0].value == "Started at +91 ms"
    workers = steps[2].table[0].value
    assert workers.to_dict("list") == {
        "Step": ["`run_rag_subtask`", "`run_rag_subtask`", "`call_tool`"],
        "Start": ["+21 ms", "+21 ms", "+22 ms"],
        "Duration": ["59 ms", "69 ms", "3.0 ms"],
        "Summary": ["s1: 2 chunks", "s3: 1 chunk", "s2: 42"],
        "Details": ["chunks: 2", "chunks: 1", ""],
    }


def test_render_trace_live_panel_is_open_and_animated() -> None:
    running = run_script(trace_panel_script, [make_event("analyze_request", 0, 5)], "running")
    starting = run_script(trace_panel_script, [], "running")

    panel = running.status[0]
    assert (panel.label, panel.icon, panel.proto.expanded) == (
        ":shimmer[Running the agent] · 1 step so far",
        "spinner",
        True,
    )
    assert [(status.label, status.icon) for status in starting.status] == [
        (":shimmer[Running the agent]", "spinner")
    ]


def test_render_trace_marks_a_failed_run() -> None:
    event = make_event("analyze_request", 0, 5, "metadata `x`", attempt=1)

    at = run_script(trace_panel_script, [event], "error")

    panel, step = at.status
    assert (panel.label, panel.icon) == (
        "Agent steps before the error · 1 step · 5.0 ms",
        ":material/error:",
    )
    assert step.markdown[0].value == r"metadata \`x\`"
    assert step.caption[0].value == "Started at +0 ms · attempt: 1"


def test_render_trace_and_render_sources_show_a_caption_for_an_empty_list() -> None:
    trace = run_script(trace_panel_script, [], "complete")
    sources = run_script(sources_panel_script, [])

    assert not trace.exception
    assert not sources.exception
    assert [caption.value for caption in trace.caption] == [
        ":material/account_tree: No agent steps were recorded for this turn."
    ]
    assert not trace.status
    assert [caption.value for caption in sources.caption] == [
        ":material/description: No context was retrieved for this turn."
    ]
    assert not sources.expander


def test_render_sources_shows_every_chunk_collapsed_with_its_details() -> None:
    sources = [
        DEMO_SOURCE,
        Source(chunk_id="notes-1", source="notes_v2.md", content="Second *chunk*, verbatim."),
    ]

    at = run_script(sources_panel_script, sources)

    assert not at.exception
    assert at.caption[0].value == ":material/library_books: Retrieved context · 2 chunks"
    first, second = at.expander
    assert (first.label, first.proto.expanded) == (
        r"\[1\] Employee handbook · p. 12 · score 0.875",
        False,
    )
    assert (second.label, second.proto.expanded) == (r"\[2\] notes\_v2.md", False)
    assert first.caption[0].value == (
        "Source: `handbook.pdf` · Title: Employee handbook · Section: Leave · Page: 12 · "
        "Score: 0.875 · Chunk: `handbook-12-3`"
    )
    assert first.text[0].value == DEMO_SOURCE.content
    assert second.caption[0].value == "Source: `notes_v2.md` · Chunk: `notes-1`"
    assert second.text[0].value == "Second *chunk*, verbatim."


# ---------------------------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------------------------


def test_group_parallel_steps_groups_overlapping_events_only() -> None:
    first = make_event("first", 0, 10)
    touching = make_event("touching", 10, 20)  # starts when `first` ends: sequential
    long_worker = make_event("long_worker", 21, 40)
    short_worker = make_event("short_worker", 25, 30)  # inside `long_worker`
    late_worker = make_event("late_worker", 35, 50)  # overlaps `long_worker` only
    chained = make_event("chained", 45, 60)  # overlaps `late_worker` only
    last = make_event("last", 61, 62)

    groups = group_parallel_steps(
        [last, chained, late_worker, short_worker, long_worker, touching, first]
    )

    assert groups == [
        [first],
        [touching],
        [long_worker, short_worker, late_worker, chained],
        [last],
    ]
    assert group_parallel_steps([]) == []


@pytest.mark.parametrize(
    ("milliseconds", "expected"),
    [
        (0, "0 ms"),
        (0.004, "0 ms"),
        (0.25, "0.25 ms"),
        (0.996, "1.0 ms"),
        (3.14159, "3.1 ms"),
        (9.96, "10 ms"),
        (52.4, "52 ms"),
        (999.4, "999 ms"),
        (999.6, "1.00 s"),
        (1250, "1.25 s"),
        (59_999, "1 min 0 s"),
        (125_000, "2 min 5 s"),
    ],
)
def test_format_duration_adapts_the_precision(milliseconds: float, expected: str) -> None:
    assert format_duration(milliseconds) == expected


def test_escape_markdown_neutralizes_markdown_syntax() -> None:
    assert escape_markdown("Plain text: 42, (a) - b.") == "Plain text: 42, (a) - b."
    assert escape_markdown(r"*a* _b_ `c` [d](e) <f> #g |h| ~i~ $j\k") == (
        r"\*a\* \_b\_ \`c\` \[d\](e) \<f\> \#g \|h\| \~i\~ \$j\\k"
    )


def test_chat_turn_as_message_keeps_only_what_the_model_saw() -> None:
    assert ChatTurn(role="user", content="Hi").as_message() == HumanMessage(content="Hi")
    assert ChatTurn(role="assistant", content="Hello").as_message() == AIMessage(content="Hello")
    assert ChatTurn(role="assistant", notice=PHASE_4_MESSAGE).as_message() is None
    assert ChatTurn(role="assistant", error=RuntimeError("x")).as_message() is None
    assert ChatTurn(role="assistant").as_message() is None


def test_settings_summary_describes_providers_models_and_top_k(settings: Settings) -> None:
    assert settings_summary(settings) == {
        ":material/smart_toy: LLM": "`fake` · scripted replies, no model",
        ":material/scatter_plot: Embeddings": "`fake` · offline vectors, no model",
        ":material/format_list_numbered: Top-k": "4",
    }
    local = settings.model_copy(
        update={"llm_provider": "ollama", "embedding_provider": "huggingface", "top_k": 8}
    )
    assert settings_summary(local) == {
        ":material/smart_toy: LLM": "`ollama` · `qwen2.5:7b-instruct`",
        ":material/lan: Ollama URL": "`http://localhost:11434`",
        ":material/scatter_plot: Embeddings": "`huggingface` · `intfloat/multilingual-e5-small`",
        ":material/format_list_numbered: Top-k": "8",
    }
