"""Streamlit entrypoint of the agentic RAG chatbot prototype.

Run it from the repository root locally, or from ``/app`` in the container::

    streamlit run src/agentic_rag/ui/app.py

The page is a chat. The conversation is kept in ``st.session_state`` as a list of ``ChatTurn``
records and replayed on every rerun. A new question goes to the main agent graph
(``agentic_rag.agent.graph.build_agent_graph``) together with the conversation so far. The app
streams the run and shows the main steps live as the nodes finish, then the answer and the
retrieved context it is grounded in. The sidebar shows the effective configuration, read-only.

Failures never end the session. A part of the agent that is planned for a later phase raises
``NotImplementedError``, and the assistant turn shows its message as a notice; any other
exception is shown with ``st.exception``. Either way the conversation is kept.

Start-up stays light and offline: this script does not import the agent graph, the LLM client,
the embedding model or the vector store. The graph is imported and built when the first
question arrives, and then shared per configuration through ``st.cache_resource``.
"""

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import streamlit as st
from pydantic import ValidationError

from agentic_rag.config import Settings, configure_logging, get_settings
from agentic_rag.rag.state import Source
from agentic_rag.tracing import TraceEvent, trace_events_from_chunk
from agentic_rag.ui.components import (
    ChatTurn,
    escape_markdown,
    render_assistant_turn,
    render_settings,
    render_trace,
    render_turn,
)

# Streamlit runs this file as `__main__`, so the logger gets its name explicitly.
logger = logging.getLogger("agentic_rag.ui.app")

HISTORY_KEY = "chat_history"
"""Session-state key of the conversation: a list of ``ChatTurn`` records, oldest first."""


@st.cache_resource(max_entries=4, show_spinner="Preparing the agent…")
def _load_agent_graph(settings_json: str, _settings: Settings) -> Any:
    """Build the main agent graph once per configuration and share it across sessions.

    Args:
        settings_json: ``_settings`` serialized as JSON. It is the cache key: Streamlit does
            not hash parameters whose name starts with an underscore.
        _settings: The settings for ``build_agent_graph``.

    Returns:
        The compiled main graph.

    Raises:
        NotImplementedError: While the main workflow is planned for a later phase. A failed
            build is not cached, so the next question tries again.
    """
    # Imported here and not at the top: importing and building the graph can load the LLM
    # client, the embedding model and the vector store, which the page start-up must not do.
    from agentic_rag.agent.graph import build_agent_graph

    return build_agent_graph(_settings)


def _run_agent(
    settings: Settings,
    history: Sequence[ChatTurn],
    on_step: Callable[[Sequence[TraceEvent]], None],
) -> ChatTurn:
    """Answer the last question of the conversation with the main agent graph.

    The run is streamed with ``stream_mode=["updates", "values"]`` as ``version="v2"`` parts.
    Every ``updates`` part carries the trace events of the node that has just finished, and
    the last ``values`` part of the root graph is the final output (``AgentOutput``). The
    step panel keeps the events the main-graph nodes recorded themselves
    (``skip_forwarded=True``): the RAG subgraph events that ``run_rag_subtask`` forwards are
    sub-steps of that worker, not main steps.

    Args:
        settings: The effective settings.
        history: The conversation, ending with the new question.
        on_step: Called with all steps so far whenever new steps arrive.

    Returns:
        The assistant turn: the answer with its sources and steps, a notice when the run
        reached a part of the agent that is planned for a later phase, or the exception of a
        failed run.
    """
    steps: list[TraceEvent] = []
    try:
        graph = _load_agent_graph(settings.model_dump_json(), settings)
        messages = [message for turn in history if (message := turn.as_message()) is not None]
        output: Mapping[str, Any] = {}
        for part in graph.stream(
            {"messages": messages}, stream_mode=["updates", "values"], version="v2"
        ):
            if part["type"] == "updates":
                new_steps = trace_events_from_chunk(part, skip_forwarded=True)
                if new_steps:
                    steps.extend(new_steps)
                    on_step(steps)
            elif part["type"] == "values" and not part["ns"]:
                output = part["data"]
        answer = str(output.get("answer") or "")
        sources = [Source.model_validate(item) for item in output.get("sources") or ()]
    except NotImplementedError as exc:
        logger.info("The agent cannot answer yet: %s", exc)
        notice = str(exc) or "This part of the agent is not built yet."
        return ChatTurn(role="assistant", notice=notice, trace=steps)
    except Exception as exc:
        logger.exception("The agent run failed")
        return ChatTurn(role="assistant", error=exc, trace=steps)
    return ChatTurn(role="assistant", content=answer, sources=sources, trace=steps)


def _describe_invalid_settings(error: ValidationError) -> str:
    """List the invalid settings by environment-variable name, as Markdown."""
    problems = []
    for item in error.errors():
        name = ".".join(str(part) for part in item["loc"]).upper() or "SETTINGS"
        problems.append(f"- `{name}`: {escape_markdown(item['msg'])}")
    return (
        "The chat cannot start because some settings are invalid. Fix these environment "
        "variables or the .env file, then restart the app.\n\n" + "\n".join(problems)
    )


def _clear_history() -> None:
    """Start a new conversation; the callback of the sidebar button."""
    st.session_state[HISTORY_KEY] = []


st.set_page_config(page_title="Agentic RAG chatbot", page_icon=":material/forum:")
st.title("Agentic RAG chatbot", anchor=False)
st.caption(
    "Prototype: a LangGraph agent answers from a local knowledge base with a local LLM. "
    "Each answer shows the steps the agent took and the retrieved context it is grounded in."
)

try:
    settings = get_settings()
except ValidationError as exc:
    st.error(
        _describe_invalid_settings(exc), title="Invalid configuration", icon=":material/error:"
    )
    st.stop()

configure_logging(settings.log_level)
history: list[ChatTurn] = st.session_state.setdefault(HISTORY_KEY, [])

with st.sidebar:
    render_settings(settings)
    st.button("Clear conversation", icon=":material/delete:", on_click=_clear_history)

question = st.chat_input("Ask a question", key="question", submit_mode="disable")

if not history and not question:
    st.caption(":material/chat: No messages yet. Ask a question below to start.")

for turn in history:
    render_turn(turn)

if question:
    user_turn = ChatTurn(role="user", content=question)
    history.append(user_turn)
    render_turn(user_turn)
    with st.chat_message("assistant"):
        # One slot for the whole turn: the live step panel while the agent runs, then the
        # finished turn, with the same layout as the turns replayed from the history.
        turn_slot = st.empty()

        def _show_steps(steps: Sequence[TraceEvent]) -> None:
            """Redraw the live step panel; the progress callback of the run."""
            with turn_slot.container():
                render_trace(steps, state="running")

        _show_steps([])
        answer_turn = _run_agent(settings, history, _show_steps)
        with turn_slot.container():
            render_assistant_turn(answer_turn)
    history.append(answer_turn)
