"""Streamlit entrypoint of the agentic RAG chatbot prototype.

Run it from the repository root locally, or from ``/app`` in the container::

    streamlit run src/agentic_rag/ui/app.py

The page is a chat. The conversation is kept in ``st.session_state`` as a list of ``ChatTurn``
records and replayed on every rerun. A new question goes to the main agent graph
(``agentic_rag.agent.graph.build_agent_graph``) together with the earlier questions that were
answered and their answers (``agent_messages``). The app streams the run and shows the main
steps live as the nodes finish, then the answer and the retrieved context it is grounded in.
The sidebar shows the effective configuration, read-only. An empty chat offers example
questions (``EXAMPLE_QUESTIONS``), one per route of the agent.

Failures never end the session. A part of the agent that is planned for a later phase raises
``agentic_rag.errors.PlannedFeatureError``, and the assistant turn shows its message as a
notice. Any other exception, a plain ``NotImplementedError`` included, is a failure. A failure
the user can fix (a missing index, an index built with other embeddings, an Ollama that cannot
be reached or lacks the model; see ``describe_failure``) is logged as a warning and explained
in the turn. Any other failure is logged with its traceback and shown with ``st.exception``.
When the user stops a run, the question still gets an assistant turn, a short notice, so the
history never keeps an unanswered question. Either way the conversation is kept. Invalid
settings, or a ``.env`` file that cannot be read, replace the chat with an error that names
the problem.

Start-up stays light and offline: this script does not import the agent graph, the LLM client,
the embedding model or the vector store. The graph is imported and built when the first
question arrives, and then shared per configuration through ``st.cache_resource``.
"""

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import streamlit as st
from pydantic import ValidationError

from agentic_rag.config import (
    Settings,
    configure_logging,
    describe_invalid_settings,
    get_settings,
)
from agentic_rag.errors import ConfigurationError, PlannedFeatureError
from agentic_rag.rag.state import Source
from agentic_rag.ui.components import (
    EXAMPLE_QUESTIONS,
    AgentStep,
    ChatTurn,
    add_reply,
    agent_messages,
    agent_steps_from_chunk,
    describe_failure,
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

EXAMPLE_KEY = "example_question"
"""Session-state key of the example question the user clicked, until the next run asks it."""


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
        PlannedFeatureError: While the main workflow is planned for a later phase. A failed
            build is not cached, so the next question tries again.
    """
    # Imported here and not at the top: importing and building the graph can load the LLM
    # client, the embedding model and the vector store, which the page start-up must not do.
    from agentic_rag.agent.graph import build_agent_graph

    return build_agent_graph(_settings)


def _run_agent(
    settings: Settings,
    history: Sequence[ChatTurn],
    steps: list[AgentStep],
    on_step: Callable[[Sequence[AgentStep]], None],
) -> ChatTurn:
    """Answer the last question of the conversation with the main agent graph.

    The graph receives the conversation as ``agent_messages(history)`` builds it. The run is
    streamed with ``stream_mode=["updates", "values"]`` as ``version="v2"`` parts. Every
    ``updates`` part carries the trace events of the node that has just finished, which
    ``agent_steps_from_chunk`` turns into main steps: the RAG subgraph events that
    ``run_rag_subtask`` forwards become sub-steps of that worker. The last ``values`` part of
    the root graph is the final output (``AgentOutput``).

    Only a ``PlannedFeatureError`` means that the run reached a part of the agent that is not
    built yet. Every other exception, a ``NotImplementedError`` from a library included, is a
    failure and is kept in the returned turn: a failure that ``describe_failure`` explains is
    logged as a warning, any other with its traceback. When the user stops the run, Streamlit
    raises its ``StopException`` (a ``BaseException``) from ``on_step``; it passes through,
    and this function does not return.

    Args:
        settings: The effective settings.
        history: The conversation, ending with the new question.
        steps: Receives the main steps as the nodes finish. The caller owns the list, so the
            steps that finished before a stop are still at hand.
        on_step: Called with all steps so far whenever new steps arrive.

    Returns:
        The assistant turn: the answer with its sources and steps, a notice when the run
        reached a part of the agent that is planned for a later phase, or the exception of a
        failed run.
    """
    try:
        graph = _load_agent_graph(settings.model_dump_json(), settings)
        output: Mapping[str, Any] = {}
        for part in graph.stream(
            {"messages": agent_messages(history)},
            stream_mode=["updates", "values"],
            version="v2",
        ):
            if part["type"] == "updates":
                new_steps = agent_steps_from_chunk(part)
                if new_steps:
                    steps.extend(new_steps)
                    on_step(steps)
            elif part["type"] == "values" and not part["ns"]:
                output = part["data"]
        answer = str(output.get("answer") or "")
        sources = [Source.model_validate(item) for item in output.get("sources") or ()]
    except PlannedFeatureError as exc:
        logger.info("The agent cannot answer yet: %s", exc)
        notice = str(exc) or "This part of the agent is not built yet."
        return ChatTurn(role="assistant", notice=notice, trace=steps)
    except Exception as exc:
        hint = describe_failure(exc, settings)
        if hint is None:
            logger.exception("The agent run failed")
        else:
            logger.warning("The agent run failed: %s: %s", hint.title, exc)
        return ChatTurn(role="assistant", error=exc, hint=hint, trace=steps)
    return ChatTurn(role="assistant", content=answer, sources=sources, trace=steps)


def _describe_settings_error(error: ValidationError | ConfigurationError) -> str:
    """Explain, as Markdown, why the settings could not be loaded.

    Args:
        error: What ``get_settings()`` raised: a ``ValidationError`` for invalid values, a
            ``ConfigurationError`` when they could not be read at all (an unreadable or
            non-UTF-8 ``.env`` file).

    Returns:
        The body of the error message that replaces the chat.
    """
    if isinstance(error, ValidationError):
        problems = [
            f"- `{name}`: {escape_markdown(problem)}"
            for name, problem in describe_invalid_settings(error)
        ]
        return (
            "The chat cannot start because some settings are invalid. Fix these environment "
            "variables or the .env file, then restart the app.\n\n" + "\n".join(problems)
        )
    return (
        "The chat cannot start because the settings could not be loaded. "
        f"{escape_markdown(str(error))}\n\nFix the problem, then restart the app."
    )


def _clear_history() -> None:
    """Start a new conversation; the callback of the sidebar button."""
    st.session_state[HISTORY_KEY] = []


def _ask_example(question: str) -> None:
    """Ask an example question in the next run; the callback of the example buttons."""
    st.session_state[EXAMPLE_KEY] = question


st.set_page_config(page_title="Agentic RAG chatbot", page_icon=":material/forum:")
st.title("Agentic RAG chatbot", anchor=False)
st.caption(
    "Prototype: a LangGraph agent answers from a local knowledge base with a local LLM. "
    "Each answer shows the steps the agent took and the retrieved context it is grounded in."
)

try:
    settings = get_settings()
except (ValidationError, ConfigurationError) as exc:
    st.error(_describe_settings_error(exc), title="Invalid configuration", icon=":material/error:")
    st.stop()

configure_logging(settings.log_level)
history: list[ChatTurn] = st.session_state.setdefault(HISTORY_KEY, [])

with st.sidebar:
    render_settings(settings)
    st.button("Clear conversation", icon=":material/delete:", on_click=_clear_history)

question = st.chat_input("Ask a question", key="question", submit_mode="disable")
question = question or st.session_state.pop(EXAMPLE_KEY, None)

if not history and not question:
    st.caption(":material/chat: No messages yet. Ask a question below, or try an example:")
    for example in EXAMPLE_QUESTIONS:
        st.button(
            example,
            icon=":material/lightbulb:",
            type="tertiary",
            on_click=_ask_example,
            args=(example,),
        )

for turn in history:
    render_turn(turn)

if question:
    user_turn = ChatTurn(role="user", content=question)
    history.append(user_turn)
    steps: list[AgentStep] = []
    reply: ChatTurn | None = None
    try:
        render_turn(user_turn)
        with st.chat_message("assistant"):
            # One slot for the whole turn: the live step panel while the agent runs, then the
            # finished turn, with the same layout as the turns replayed from the history. A
            # redraw replaces the slot's container with a new one, which keeps the elements of
            # the old one that it does not overwrite; render_trace therefore never draws fewer
            # elements in a place than an earlier redraw of the same run.
            turn_slot = st.empty()

            def _show_steps(steps_so_far: Sequence[AgentStep]) -> None:
                """Redraw the live step panel; the progress callback of the run."""
                with turn_slot.container():
                    render_trace(steps_so_far, state="running")

            _show_steps(steps)
            reply = _run_agent(settings, history, steps, _show_steps)
            with turn_slot.container():
                render_assistant_turn(reply)
    except Exception as exc:
        # _run_agent handles the failures of the run, so this is a failure of the page itself:
        # close the question with it, and let Streamlit show it.
        if reply is None:
            reply = ChatTurn(role="assistant", error=exc, trace=steps)
        raise
    finally:
        # This also runs when the user stops the run. Streamlit then raises StopException (a
        # BaseException) at the next Streamlit call, and again at every call after it, reads
        # and writes of st.session_state included. So this block only changes the history list
        # it already holds, and renders nothing: the next rerun shows the turn.
        if reply is None:
            reply = ChatTurn(role="assistant", stopped=True, trace=steps)
        add_reply(history, user_turn, reply)
