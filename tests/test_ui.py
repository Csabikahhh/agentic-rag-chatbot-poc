"""Tests of the Streamlit UI: the entrypoint under AppTest and the components on their own.

The app runs headless and offline with ``streamlit.testing.v1.AppTest`` in fake mode. The main
graph module ``agentic_rag.agent.graph`` is replaced in ``sys.modules`` by stand-ins, so the
tests do not depend on the real module (a stub until Phase 4): a builder that raises
``PlannedFeatureError`` like the stub, one that raises an unexpected error, and builders that
return small real LangGraph graphs on the project's state contracts. Some stand-in nodes press
Stop the way the browser does, through the stop request of the script runner.
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
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send
from streamlit.runtime.scriptrunner import get_script_run_ctx
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
from agentic_rag.errors import planned
from agentic_rag.rag.state import Source
from agentic_rag.tracing import TraceEvent, epoch_now, traced
from agentic_rag.ui.components import (
    ChatTurn,
    add_reply,
    agent_messages,
    escape_dollar_signs,
    escape_markdown,
    format_duration,
    group_parallel_steps,
    settings_summary,
)
from conftest import heavy_modules

APP_PATH = Path(agentic_rag.ui.__file__).with_name("app.py")
GRAPH_MODULE = "agentic_rag.agent.graph"
HISTORY_KEY = "chat_history"
PHASE_4_MESSAGE = str(planned("agentic_rag.agent.graph.build_agent_graph", 4))
NODE_STUB_MESSAGE = str(planned("agentic_rag.agent.nodes.plan_subtasks", 4))
STOPPED_NOTICE = "Stopped before an answer was produced. Ask again to get one."
# The shared list of conftest.py, except Streamlit: the UI is where it belongs.
UI_HEAVY_MODULES = heavy_modules(allow={"streamlit"})
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
    """Clear Streamlit's process-wide resource cache before and after every test.

    ``st.cache_resource`` outlives an AppTest run, so a graph cached in one test would answer
    in the next one. (The logging setup that the app's ``configure_logging()`` call changes is
    restored by the autouse fixture of ``conftest.py``.)
    """
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


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
        if fullname.partition(".")[0] in UI_HEAVY_MODULES:
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
        if name.partition(".")[0] in UI_HEAVY_MODULES:
            monkeypatch.delitem(sys.modules, name)
    blocker = _HeavyImportBlocker()
    monkeypatch.setattr(sys, "meta_path", [blocker, *sys.meta_path])
    return blocker.attempts


def install_graph_module(
    monkeypatch: pytest.MonkeyPatch, build: Callable[[Settings], Any]
) -> list[Settings]:
    """Replace ``agentic_rag.agent.graph`` with a stand-in whose builder delegates to build.

    Returns:
        The settings of every ``build_agent_graph`` call, in call order.
    """
    calls: list[Settings] = []

    def build_agent_graph(settings: Settings) -> Any:
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


def message_texts(messages: Sequence[BaseMessage]) -> list[str]:
    """Return ``"<type>: <content>"`` for every message, for example ``"human: Hi"``."""
    return [f"{message.type}: {message.content}" for message in messages]


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


def press_stop() -> None:
    """Do what the browser's Stop button does while the script runs.

    The session asks the script runner to stop, and Streamlit raises ``StopException`` at the
    next Streamlit call of the script. Call it from a node of a single-task step: LangGraph
    runs such a step in the calling thread, the script thread, which holds the run context.
    """
    ctx = get_script_run_ctx()
    assert ctx is not None, "press_stop() must run in the script thread"
    assert ctx.script_requests is not None
    ctx.script_requests.request_stop()


# Scripts for AppTest.from_function: the body runs as a page, so it imports what it uses.


def trace_panel_script(events: list, state: str) -> None:
    """Render one step panel."""
    from agentic_rag.ui.components import render_trace

    render_trace(events, state=state)


def sources_panel_script(sources: list) -> None:
    """Render one retrieved-context panel."""
    from agentic_rag.ui.components import render_sources

    render_sources(sources)


def reply_script(turn: object) -> None:
    """Render the reply of one assistant turn (a ``ChatTurn``)."""
    from agentic_rag.ui.components import render_reply

    render_reply(turn)


# ---------------------------------------------------------------------------------------------
# Stand-in graph builders
# ---------------------------------------------------------------------------------------------


def raise_planned(settings: Settings) -> Any:
    """Behave like the Phase 4 stub of ``build_agent_graph``."""
    raise planned(f"{GRAPH_MODULE}.build_agent_graph", 4)


def raise_runtime_error(settings: Settings) -> Any:
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
    raise planned("agentic_rag.agent.nodes.plan_subtasks", 4)


@traced
def call_async_only_tool(state: AgentState) -> dict[str, Any]:
    """Fail like ``.invoke()`` on a LangChain tool that has only an async implementation."""
    raise NotImplementedError("StructuredTool does not support sync invocation.")


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


@traced
def answer_ok(state: AgentState) -> dict[str, Any]:
    """Answer every question with "ok"."""
    return {"answer": "ok", "sources": [], "messages": [AIMessage(content="ok")]}


def build_demo_graph(settings: Settings) -> CompiledStateGraph:
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


def build_linear_graph(*nodes: Callable[[AgentState], Any]) -> CompiledStateGraph:
    """Build a graph on the state contracts that runs the nodes one after the other.

    Each node is registered under its function name, so every step has a single task, which
    LangGraph runs in the calling thread.
    """
    builder = StateGraph(AgentState, input_schema=AgentInput, output_schema=AgentOutput)
    previous = START
    for node in nodes:
        builder.add_node(node.__name__, node)
        builder.add_edge(previous, node.__name__)
        previous = node.__name__
    builder.add_edge(previous, END)
    return builder.compile()


def build_partial_graph(settings: Settings) -> CompiledStateGraph:
    """Build a graph whose second node is still a stub raising PlannedFeatureError."""
    return build_linear_graph(analyze_request, plan_subtasks_stub)


# ---------------------------------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------------------------------


def test_app_starts_offline_in_fake_mode_without_loading_heavy_libraries(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, heavy_imports: list[str]
) -> None:
    # Import both modules afresh during the run, so their imports go through the blocker.
    # monkeypatch puts the old components module back afterwards, in sys.modules and as the
    # attribute of its package, which the fresh import replaces.
    monkeypatch.delitem(sys.modules, GRAPH_MODULE, raising=False)
    monkeypatch.delitem(sys.modules, "agentic_rag.ui.components")
    monkeypatch.setattr(agentic_rag.ui, "components", agentic_rag.ui.components)

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


def test_question_shows_the_planned_feature_notice_and_keeps_the_history(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = install_graph_module(monkeypatch, raise_planned)
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
    assert [call.llm_provider for call in calls] == ["fake", "fake"]


def test_planned_feature_inside_the_run_keeps_the_steps_before_it(
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


def test_plain_not_implemented_error_is_a_failure_with_its_traceback(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    install_graph_module(
        monkeypatch, lambda settings: build_linear_graph(analyze_request, call_async_only_tool)
    )
    at = start_app()

    ask(at, "What is 6 times 7?")

    assistant = at.chat_message[1]
    assert [error.value for error in assistant.exception] == [
        "StructuredTool does not support sync invocation."
    ]
    assert not assistant.info  # Not mistaken for a part that is planned for a later phase.
    assert assistant.status[0].label.startswith("Agent steps before the error · 1 step · ")
    (record,) = [record for record in caplog.records if record.name == "agentic_rag.ui.app"]
    assert (record.levelno, record.getMessage()) == (logging.ERROR, "The agent run failed")
    assert record.exc_info is not None
    assert type(record.exc_info[1]) is NotImplementedError


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
    # The RAG subgraph event that run_rag_subtask forwards is not a main step. The workers in
    # between may or may not be grouped as parallel, depending on whether their run times
    # overlapped (see group_parallel_steps).
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


def test_stopped_run_closes_its_question_and_later_runs_leave_it_out(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[list[str]] = []

    @traced
    def analyze_request(state: AgentState) -> dict[str, Any]:
        """Record the conversation the agent receives; press Stop during the first run."""
        seen.append(message_texts(state["messages"]))
        if len(seen) == 1:
            press_stop()
        return {"question": str(state["messages"][-1].content), "intent": "direct"}

    install_graph_module(
        monkeypatch, lambda settings: build_linear_graph(analyze_request, answer_ok)
    )
    at = start_app()

    ask(at, "First question")

    # The stop came at the live update after analyze_request; the question was closed anyway.
    assert not at.exception
    question, reply = at.session_state[HISTORY_KEY]
    assert (question.content, reply.role, reply.stopped, reply.content) == (
        "First question",
        "assistant",
        True,
        "",
    )
    assert [event.node for event in reply.trace] == ["analyze_request"]

    at.run()  # The next rerun shows the closed turn.

    assistant = at.chat_message[1]
    assert [notice.value for notice in assistant.info] == [STOPPED_NOTICE]
    assert assistant.status[0].label.startswith("Agent steps before the stop · 1 step · ")
    assert assistant.status[0].icon == ":material/stop_circle:"

    ask(at, "Second question")

    assert not at.exception
    # The stopped question is not sent again, so the agent never gets two user messages in a row.
    assert seen == [["human: First question"], ["human: Second question"]]
    assert [(turn.role, turn.content) for turn in at.session_state[HISTORY_KEY]] == [
        ("user", "First question"),
        ("assistant", ""),
        ("user", "Second question"),
        ("assistant", "ok"),
    ]


def test_stop_after_the_answer_was_computed_keeps_the_answer(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def stop_after_answering(state: AgentState) -> None:
        """Press Stop in the last node; untraced, so the stop lands on the final render."""
        press_stop()

    install_graph_module(
        monkeypatch, lambda settings: build_linear_graph(answer_ok, stop_after_answering)
    )
    at = start_app()

    ask(at, "First question")

    assert not at.exception
    question, reply = at.session_state[HISTORY_KEY]
    assert (question.content, reply.content, reply.stopped) == ("First question", "ok", False)

    at.run()

    assert "ok" in [markdown.value for markdown in at.chat_message[1].markdown]
    assert not at.chat_message[1].info


def test_failure_of_the_page_itself_closes_the_question_with_the_error(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_render_trace(*args: object, **kwargs: object) -> None:
        raise RuntimeError("the step panel broke")

    install_graph_module(monkeypatch, build_demo_graph)
    # Every run of the app imports render_trace from the module in sys.modules again.
    components = sys.modules["agentic_rag.ui.components"]
    monkeypatch.setattr(components, "render_trace", broken_render_trace)
    at = start_app()

    ask(at, "First question")

    assert [error.value for error in at.exception] == ["the step panel broke"]
    question, reply = at.session_state[HISTORY_KEY]
    assert (question.content, reply.stopped, str(reply.error)) == (
        "First question",
        False,
        "the step panel broke",
    )


def test_clear_conversation_starts_over(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_graph_module(monkeypatch, raise_planned)
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
    (error,) = at.error
    assert error.proto.title == "Invalid configuration"
    assert error.value == (
        "The chat cannot start because some settings are invalid. Fix these environment "
        "variables or the .env file, then restart the app.\n\n"
        "- `TOP_K`: Input should be greater than or equal to 1 (got '0')"
    )
    assert not at.chat_input


def test_unreadable_env_file_shows_an_error_instead_of_the_chat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # What `echo LLM_PROVIDER=fake > .env` writes in Windows PowerShell 5.1: UTF-16 with a BOM.
    (tmp_path / ".env").write_text("LLM_PROVIDER=fake\n", encoding="utf-16")
    # The app reads .env from the working directory; conftest turns .env loading off.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(Settings.model_config, "env_file", ".env")

    at = start_app()

    assert not at.exception
    (error,) = at.error
    assert error.proto.title == "Invalid configuration"
    assert error.value.startswith(
        "The chat cannot start because the settings could not be loaded. "
        + escape_markdown(f"{Path('.env').absolute()} is not valid UTF-8")
    )
    assert error.value.endswith("\n\nFix the problem, then restart the app.")
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


@pytest.mark.parametrize(
    ("state", "title", "icon"),
    [
        ("error", "Agent steps before the error", ":material/error:"),
        ("stopped", "Agent steps before the stop", ":material/stop_circle:"),
    ],
)
def test_render_trace_marks_a_run_that_did_not_finish(state: str, title: str, icon: str) -> None:
    event = make_event("analyze_request", 0, 5, "metadata `x`", attempt=1)

    at = run_script(trace_panel_script, [event], state)

    panel, step = at.status
    assert (panel.label, panel.icon) == (f"{title} · 1 step · 5.0 ms", icon)
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


def test_render_reply_escapes_the_dollar_signs_of_the_answer_outside_code() -> None:
    answer = "The fee is $25 and the late fee is $40; `echo $HOME` stays as it is."

    at = run_script(reply_script, ChatTurn(role="assistant", content=answer))

    assert not at.exception
    assert [markdown.value for markdown in at.markdown] == [
        r"The fee is \$25 and the late fee is \$40; `echo $HOME` stays as it is."
    ]


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


def test_group_parallel_steps_sees_time_only_not_langgraph_steps() -> None:
    # Two Send workers of one planning round whose run times did not overlap, as when a fast
    # pure-Python tool returns before the next worker starts, show as sequential steps.
    first_tool = make_event("call_tool", 20, 21, "s1: 42")
    second_tool = make_event("call_tool", 21.5, 22, "s2: 7")

    assert group_parallel_steps([second_tool, first_tool]) == [[first_tool], [second_tool]]


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


@pytest.mark.parametrize(
    ("markdown", "expected"),
    [
        pytest.param(
            "The fee is $25 and the late fee is $40.",
            r"The fee is \$25 and the late fee is \$40.",
            id="currency",
        ),
        pytest.param("$$x^2$$", r"\$\$x^2\$\$", id="math-shows-as-source"),
        pytest.param(r"\$5 and \\$6", r"\$5 and \\\$6", id="already-escaped"),
        pytest.param(r"\`$5\`", r"\`\$5\`", id="escaped-backticks-are-not-code"),
        pytest.param("Run `echo $HOME` for $5.", r"Run `echo $HOME` for \$5.", id="code-span"),
        pytest.param("``a ` $b`` and $c", r"``a ` $b`` and \$c", id="double-backtick-span"),
        pytest.param("A lone ` and $5", r"A lone ` and \$5", id="unmatched-backtick"),
        pytest.param("Use `a\nb $c` for $5", "Use `a\nb $c` for \\$5", id="span-across-lines"),
        pytest.param(
            "- `a` costs $1\n- `b` costs $2",
            "- `a` costs \\$1\n- `b` costs \\$2",
            id="code-spans-in-list-items",
        ),
        pytest.param(
            "- unmatched ` here\n- $3 and `x`",
            "- unmatched ` here\n- \\$3 and `x`",
            id="span-does-not-reach-the-next-item",
        ),
        pytest.param("```py``` costs $5", "```py``` costs \\$5", id="inline-triple-backticks"),
        pytest.param(
            "$5\n```bash\necho $HOME\n```\n$6",
            "\\$5\n```bash\necho $HOME\n```\n\\$6",
            id="backtick-fence",
        ),
        pytest.param("~~~\n$x$\n~~~\n$y", "~~~\n$x$\n~~~\n\\$y", id="tilde-fence"),
        pytest.param(
            "````\n```\n$x\n````\n$y", "````\n```\n$x\n````\n\\$y", id="longer-closing-fence"
        ),
        pytest.param(
            "```\n$x$\n``` not a closing fence\n$y$",
            "```\n$x$\n``` not a closing fence\n$y$",
            id="unclosed-fence-runs-to-the-end",
        ),
        pytest.param(
            "1. Run:\n   ```sh\n   echo $HOME\n   ```\n2. Pay $5",
            "1. Run:\n   ```sh\n   echo $HOME\n   ```\n2. Pay \\$5",
            id="fence-in-a-list-item",
        ),
        pytest.param(
            "Example:\n\n    total = $5 + $6\n\nPay $7",
            "Example:\n\n    total = $5 + $6\n\nPay \\$7",
            id="indented-code",
        ),
        pytest.param("\t$x = 1$", "\t$x = 1$", id="tab-indented-code"),
        pytest.param(
            "# Price $5\n    $x = 1$", "# Price \\$5\n    $x = 1$", id="code-after-heading"
        ),
        pytest.param("Pay $5\n    or $6", "Pay \\$5\n    or \\$6", id="paragraph-continuation"),
        pytest.param(
            "- Gross: $10\n\n    net $8 or $9",
            "- Gross: \\$10\n\n    net \\$8 or \\$9",
            id="indented-prose-in-a-list",
        ),
        pytest.param(
            "Cost $5\r\n```\r\n$x\r\n```\r\n",
            "Cost \\$5\r\n```\r\n$x\r\n```\r\n",
            id="crlf-line-endings",
        ),
        pytest.param("", "", id="empty"),
    ],
)
def test_escape_dollar_signs_escapes_the_prose_and_leaves_code_alone(
    markdown: str, expected: str
) -> None:
    assert escape_dollar_signs(markdown) == expected


def test_chat_turn_as_message_keeps_only_what_the_model_saw() -> None:
    assert ChatTurn(role="user", content="Hi").as_message() == HumanMessage(content="Hi")
    assert ChatTurn(role="assistant", content="Hello").as_message() == AIMessage(content="Hello")
    assert ChatTurn(role="assistant", notice=PHASE_4_MESSAGE).as_message() is None
    assert ChatTurn(role="assistant", stopped=True).as_message() is None
    assert ChatTurn(role="assistant", error=RuntimeError("x")).as_message() is None
    assert ChatTurn(role="assistant").as_message() is None


def test_agent_messages_keep_the_answered_questions_and_the_new_one() -> None:
    history = [
        ChatTurn(role="user", content="answered"),
        ChatTurn(role="assistant", content="answer"),
        ChatTurn(role="user", content="planned"),
        ChatTurn(role="assistant", notice=PHASE_4_MESSAGE),
        ChatTurn(role="user", content="stopped"),
        ChatTurn(role="assistant", stopped=True),
        ChatTurn(role="user", content="failed"),
        ChatTurn(role="assistant", error=RuntimeError("x")),
        # Left without a reply by a stopped run that has not finished yet.
        ChatTurn(role="user", content="orphan"),
        ChatTurn(role="user", content="new question"),
    ]

    assert message_texts(agent_messages(history)) == [
        "human: answered",
        "ai: answer",
        "human: new question",
    ]
    assert message_texts(agent_messages(history[-1:])) == ["human: new question"]
    assert agent_messages([]) == []


def test_add_reply_puts_the_reply_right_after_its_question() -> None:
    first = ChatTurn(role="user", content="Same question")
    answer = ChatTurn(role="assistant", content="Answer")
    again = ChatTurn(role="user", content="Same question")  # Equal to `first`, another turn.
    newer = ChatTurn(role="user", content="Newer question")
    history = [first]

    add_reply(history, first, answer)
    assert [id(turn) for turn in history] == [id(first), id(answer)]

    # A stopped run that winds down after the user asked a newer question.
    history.extend([again, newer])
    stopped = ChatTurn(role="assistant", stopped=True)
    add_reply(history, again, stopped)

    expected = [first, answer, again, stopped, newer]
    assert [id(turn) for turn in history] == [id(turn) for turn in expected]


def test_settings_summary_describes_providers_models_and_top_k(settings: Settings) -> None:
    assert settings_summary(settings) == {
        ":material/smart_toy: LLM": "`fake` · scripted replies, no model",
        ":material/scatter_plot: Embeddings": "`fake` · offline vectors, no model",
        ":material/format_list_numbered: Top-k": "4",
    }
    local = Settings.model_validate(
        {
            **settings.model_dump(),
            "llm_provider": "ollama",
            "embedding_provider": "huggingface",
            "top_k": 8,
        }
    )
    assert settings_summary(local) == {
        ":material/smart_toy: LLM": "`ollama` · `qwen2.5:7b-instruct`",
        ":material/lan: Ollama URL": "`http://localhost:11434`",
        ":material/scatter_plot: Embeddings": "`huggingface` · `intfloat/multilingual-e5-small`",
        ":material/format_list_numbered: Top-k": "8",
    }
