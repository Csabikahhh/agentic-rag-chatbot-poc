"""Building blocks of the Streamlit chat UI.

The entrypoint (``app.py``) keeps the conversation in ``st.session_state`` as a list of
:class:`ChatTurn` records and replays it on every rerun with :func:`render_turn`. Every
assistant turn has the same three parts, in this order (:func:`render_assistant_turn`):

1. :func:`render_trace`: the main steps the agent took (node name, duration, summary) as a
   collapsible step timeline. Steps whose time windows overlap, which are the parallel
   ``Send`` workers of one planning round, are grouped into one "in parallel" step.
2. :func:`render_reply`: the answer, a notice when the run reached a part of the agent that is
   planned for a later phase, or the exception when the run failed.
3. :func:`render_sources`: the RAG result: every retrieved chunk with its source, title, page,
   section, score and text, collapsed by default.

:func:`render_settings` shows the effective configuration in the sidebar.

:class:`ChatTurn` lives in this module and not in the script because Streamlit executes the
script again on every rerun: a class defined there would be a new class on each run, while the
records in the session state would still be instances of the old one.

Text that comes from data (node names, summaries, titles, error messages) is escaped before it
is rendered as Markdown, and chunk text is shown verbatim with ``st.text``. Only the answer is
rendered as Markdown, because the model writes Markdown on purpose.
"""

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, ConfigDict, Field
from streamlit.delta_generator import DeltaGenerator

from agentic_rag import __version__
from agentic_rag.config import Settings
from agentic_rag.rag.state import Source
from agentic_rag.tracing import TraceEvent

__all__ = [
    "ChatTurn",
    "Role",
    "TraceState",
    "escape_markdown",
    "format_duration",
    "group_parallel_steps",
    "render_assistant_turn",
    "render_reply",
    "render_settings",
    "render_sources",
    "render_trace",
    "render_turn",
    "settings_summary",
]

Role = Literal["user", "assistant"]
"""Author of a chat turn; also the name passed to ``st.chat_message``."""

TraceState = Literal["running", "complete", "error"]
"""State of the step panel: the run is in progress, has finished, or has failed."""

# Icon of the step panel per state. The panel is an `st.expander`, not an `st.status`: in
# Streamlit 1.64 every `st.status` sleeps 50 ms when it is created, which would delay every
# live update of the panel and every assistant turn replayed on a rerun.
_PANEL_ICONS: dict[TraceState, str] = {
    "running": "spinner",
    "complete": ":material/account_tree:",
    "error": ":material/error:",
}

# Characters that start inline Markdown (emphasis, code, links, HTML, tables, strikethrough,
# LaTeX) or a heading. Other line-start syntax (list markers, quotes) is left alone: labels
# render inline only, and in a body it can at most turn a line into a list item or a quote.
_MARKDOWN_SPECIALS = re.compile(r"([\\`*_\[\]<>#|~$])")
_BACKTICK_RUN = re.compile(r"`+")


class ChatTurn(BaseModel):
    """One message of the conversation, as kept in ``st.session_state``.

    A user turn holds the question in ``content``. An assistant turn holds the outcome of one
    agent run: the answer in ``content`` (Markdown), the main steps in ``trace`` and the
    retrieved chunks in ``sources``. When the run reached a part of the agent that is planned
    for a later phase, ``notice`` holds that message instead of an answer; when the run failed,
    ``error`` holds the exception. Turns are immutable values.

    Attributes:
        role: Author of the turn.
        content: The question, or the answer in Markdown; empty when there is no answer.
        trace: The main steps of the run, as the graph nodes recorded them.
        sources: The retrieved chunks behind the answer, in citation order.
        notice: Message of the ``NotImplementedError`` that stopped the run.
        error: The exception that made the run fail.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    role: Role
    content: str = ""
    trace: list[TraceEvent] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    notice: str | None = None
    error: Exception | None = None

    def as_message(self) -> BaseMessage | None:
        """Return the turn as a message of the conversation history the agent receives.

        Returns:
            A ``HumanMessage`` for a user turn, an ``AIMessage`` for an assistant turn with an
            answer, and None for an assistant turn without one (a notice or an error): the
            model did not write those, so they are not part of its history.
        """
        if self.role == "user":
            return HumanMessage(content=self.content)
        if self.content and self.notice is None and self.error is None:
            return AIMessage(content=self.content)
        return None


def render_turn(turn: ChatTurn) -> None:
    """Render one turn of the conversation as a chat message.

    A user turn shows the question as plain text; an assistant turn shows its three parts
    (see :func:`render_assistant_turn`).

    Args:
        turn: The turn to render.
    """
    with st.chat_message(turn.role):
        if turn.role == "user":
            st.text(turn.content)
        else:
            render_assistant_turn(turn)


def render_assistant_turn(turn: ChatTurn) -> None:
    """Render the parts of an assistant turn into the current container.

    The step panel, the reply and the retrieved context, in this order. The two panels are
    shown even when they are empty (as a short caption), so every assistant turn has the same
    layout.

    Args:
        turn: An assistant turn.
    """
    render_trace(turn.trace, state="error" if turn.error is not None else "complete")
    render_reply(turn)
    render_sources(turn.sources)


def render_reply(turn: ChatTurn) -> None:
    """Render the reply of an assistant turn.

    By priority: the exception of a failed run with its traceback (``st.exception``), the
    notice of a part of the agent that is not built yet, the answer as Markdown, or a warning
    when the run finished without an answer.

    Args:
        turn: An assistant turn.
    """
    if turn.error is not None:
        st.exception(turn.error)
    elif turn.notice is not None:
        st.info(
            f"The agent cannot answer yet: {escape_markdown(turn.notice)}\n\n"
            "The chat interface works and the conversation is kept.",
            title="Not available yet",
            icon=":material/construction:",
        )
    elif turn.content:
        st.markdown(turn.content)
    else:
        st.warning("The agent finished without an answer.", icon=":material/help:")


def render_trace(events: Sequence[TraceEvent], *, state: TraceState = "complete") -> None:
    """Render the main steps of an agent run as a collapsible step timeline.

    The panel is a compact expander whose label sums up the run (the number of steps and the
    wall-clock time) and whose icon shows the state. Inside, the steps follow each other in
    the order they started, each with its node name and duration in the label and its
    summary, start offset and metadata in the body. Steps whose time windows overlap (the
    ``Send`` workers of one planning round) form one "in parallel" step with a table of the
    workers, so concurrent work stays readable (see :func:`group_parallel_steps`).

    Pass the events the nodes recorded themselves, as
    ``trace_events_from_chunk(chunk, skip_forwarded=True)`` returns them: an event that a node
    forwarded from a subgraph overlaps the node's own event and would be grouped with it.

    Args:
        events: Trace events in any order; may be empty.
        state: ``"running"`` while the run is in progress: the panel is open, with a spinner
            and an animated label. ``"complete"`` after a run and ``"error"`` after a failed
            run: the panel is collapsed, or replaced by a short caption when there are no
            events.
    """
    if not events and state != "running":
        st.caption(":material/account_tree: No agent steps were recorded for this turn.")
        return
    panel = st.expander(
        _trace_label(events, state),
        expanded=state == "running",
        icon=_PANEL_ICONS[state],
        type="compact",
    )
    origin = min(event.started_at for event in events) if events else 0.0
    for group in group_parallel_steps(events):
        if len(group) == 1:
            _render_step(panel, group[0], origin)
        else:
            _render_parallel_steps(panel, group, origin)


def render_sources(sources: Sequence[Source]) -> None:
    """Render the retrieved context behind an answer, one collapsed expander per chunk.

    The label of each expander shows the citation number, the title (or the source when the
    chunk has no title), the page and the score. The body shows every known detail (source,
    title, section, page, score, chunk id) and the chunk text verbatim.

    Args:
        sources: Retrieved chunks in citation order; may be empty.
    """
    if not sources:
        st.caption(":material/description: No context was retrieved for this turn.")
        return
    st.caption(f":material/library_books: Retrieved context · {_count(len(sources), 'chunk')}")
    for number, source in enumerate(sources, start=1):
        with st.expander(_source_label(number, source)):
            st.caption(_source_details(source))
            if source.content:
                st.text(source.content)
            else:
                st.caption("This chunk has no text.")


def render_settings(settings: Settings) -> None:
    """Render the effective configuration, read-only, for the sidebar.

    Args:
        settings: The effective settings, as returned by ``get_settings()``.
    """
    st.subheader("Configuration", anchor=False)
    st.table(settings_summary(settings), border="horizontal", width="content")
    st.caption(
        "Read-only. Change it with environment variables or a .env file (see .env.example), "
        "then restart the app."
    )
    st.caption(f"agentic-rag {__version__}")


def settings_summary(settings: Settings) -> dict[str, str]:
    """Describe the configuration that the sidebar shows.

    Args:
        settings: The effective settings.

    Returns:
        Row labels (with a Material icon) mapped to Markdown values: the LLM provider and
        model, the Ollama URL (only for the ``ollama`` provider), the embedding provider and
        model, and the retrieval depth (top-k).
    """
    summary: dict[str, str] = {}
    if settings.llm_provider == "ollama":
        summary[":material/smart_toy: LLM"] = f"`ollama` · {_inline_code(settings.ollama_model)}"
        summary[":material/lan: Ollama URL"] = _inline_code(settings.ollama_base_url)
    else:
        summary[":material/smart_toy: LLM"] = "`fake` · scripted replies, no model"
    if settings.embedding_provider == "huggingface":
        summary[":material/scatter_plot: Embeddings"] = (
            f"`huggingface` · {_inline_code(settings.embedding_model)}"
        )
    else:
        summary[":material/scatter_plot: Embeddings"] = "`fake` · offline vectors, no model"
    summary[":material/format_list_numbered: Top-k"] = str(settings.top_k)
    return summary


def group_parallel_steps(events: Sequence[TraceEvent]) -> list[list[TraceEvent]]:
    """Order trace events by start time and group the ones that ran at the same time.

    Two events overlap when one starts before the other ends. A group is a maximal run of
    events that overlap each other, directly or through other events of the group: the
    workers of one ``Send`` fan-out form one group, and every sequential step of the main
    workflow is a group of its own. Events that only touch (one starts exactly when the other
    ends) are sequential.

    Args:
        events: Trace events in any order.

    Returns:
        The groups in chronological order. Inside a group, the events are ordered by start
        time, then by end time.
    """
    groups: list[list[TraceEvent]] = []
    group_end = 0.0
    for event in sorted(events, key=lambda item: (item.started_at, item.ended_at)):
        if groups and event.started_at < group_end:
            groups[-1].append(event)
            group_end = max(group_end, event.ended_at)
        else:
            groups.append([event])
            group_end = event.ended_at
    return groups


def format_duration(milliseconds: float) -> str:
    """Format a duration with a precision that suits its size.

    Args:
        milliseconds: A duration in milliseconds; not negative.

    Returns:
        For example ``"0 ms"``, ``"0.25 ms"``, ``"3.1 ms"``, ``"52 ms"``, ``"1.25 s"`` or
        ``"2 min 5 s"``.
    """
    # The thresholds sit where rounding would carry over into the next format.
    if milliseconds < 0.005:
        return "0 ms"
    if milliseconds < 0.995:
        return f"{milliseconds:.2f} ms"
    if milliseconds < 9.95:
        return f"{milliseconds:.1f} ms"
    if milliseconds < 999.5:
        return f"{milliseconds:.0f} ms"
    seconds = milliseconds / 1000.0
    if seconds < 59.995:
        return f"{seconds:.2f} s"
    minutes, rest = divmod(round(seconds), 60)
    return f"{minutes} min {rest} s"


def escape_markdown(text: str) -> str:
    """Escape the characters that Streamlit's Markdown would interpret in a line of text.

    Use it for text that comes from data (node names, summaries, titles, error messages) and
    is shown by an element or label that renders Markdown.

    Args:
        text: Plain text.

    Returns:
        The text with a backslash in front of every Markdown special character.
    """
    return _MARKDOWN_SPECIALS.sub(r"\\\1", text)


def _render_step(parent: DeltaGenerator, event: TraceEvent, origin: float) -> None:
    """Render one step of the timeline: node and duration in the label, details in the body."""
    step = parent.expander(
        f"{_inline_code(event.node)} · {format_duration(event.duration_ms)}",
        expanded=True,
        icon=":material/check_circle:",
        type="step",
    )
    if event.summary:
        step.markdown(escape_markdown(event.summary))
    # A step without content would end the timeline, so the details line is always present.
    details = [f"Started at +{_offset(event, origin)}", *_metadata_items(event.metadata)]
    step.caption(" · ".join(details))


def _render_parallel_steps(
    parent: DeltaGenerator, group: Sequence[TraceEvent], origin: float
) -> None:
    """Render steps that ran at the same time as one timeline step with a table of them."""
    with_details = any(event.metadata for event in group)
    rows: list[dict[str, str]] = []
    for event in group:
        row = {
            "Step": _inline_code(event.node),
            "Start": f"+{_offset(event, origin)}",
            "Duration": format_duration(event.duration_ms),
            "Summary": escape_markdown(event.summary),
        }
        if with_details:
            row["Details"] = " · ".join(_metadata_items(event.metadata))
        rows.append(row)
    step = parent.expander(
        f"{len(group)} steps in parallel · {format_duration(_wall_ms(group))}",
        expanded=True,
        icon=":material/call_split:",
        type="step",
    )
    step.table(rows, hide_index=True, border="horizontal")


def _trace_label(events: Sequence[TraceEvent], state: TraceState) -> str:
    """Return the label of the step panel: what is running, or how many steps took how long."""
    if state == "running":
        progress = f" · {_count(len(events), 'step')} so far" if events else ""
        return f":shimmer[Running the agent]{progress}"
    summary = f"{_count(len(events), 'step')} · {format_duration(_wall_ms(events))}"
    if state == "error":
        return f"Agent steps before the error · {summary}"
    return f"Agent steps · {summary}"


def _wall_ms(events: Sequence[TraceEvent]) -> float:
    """Return the time from the first start to the last end, in milliseconds."""
    started = min(event.started_at for event in events)
    ended = max(event.ended_at for event in events)
    return (ended - started) * 1000.0


def _offset(event: TraceEvent, origin: float) -> str:
    """Return the start of an event relative to the start of the run, formatted."""
    return format_duration(max(event.started_at - origin, 0.0) * 1000.0)


def _metadata_items(metadata: Mapping[str, Any]) -> list[str]:
    """Format event metadata as escaped ``key: value`` items."""
    return [
        f"{escape_markdown(str(key))}: {escape_markdown(_format_value(value))}"
        for key, value in metadata.items()
    ]


def _format_value(value: Any) -> str:
    """Format one metadata value: strings as they are, anything else as compact JSON."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _inline_code(text: str) -> str:
    """Wrap text in a Markdown code span, whatever backticks it contains."""
    longest_run = max((len(run) for run in _BACKTICK_RUN.findall(text)), default=0)
    fence = "`" * (longest_run + 1)
    padding = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{fence}{padding}{text}{padding}{fence}"


def _count(number: int, noun: str) -> str:
    """Return ``"1 step"``, ``"2 steps"`` and so on."""
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def _source_label(number: int, source: Source) -> str:
    """Return the expander label of a chunk: citation number, title or source, page, score."""
    parts = [f"\\[{number}\\] {escape_markdown(source.title or source.source)}"]
    if source.page is not None:
        parts.append(f"p. {source.page}")
    if source.score is not None:
        parts.append(f"score {source.score:.3f}")
    return " · ".join(parts)


def _source_details(source: Source) -> str:
    """Return every known detail of a chunk on one line."""
    details = [f"Source: {_inline_code(source.source)}"]
    if source.title:
        details.append(f"Title: {escape_markdown(source.title)}")
    if source.section:
        details.append(f"Section: {escape_markdown(source.section)}")
    if source.page is not None:
        details.append(f"Page: {source.page}")
    if source.score is not None:
        details.append(f"Score: {source.score:.3f}")
    details.append(f"Chunk: {_inline_code(source.chunk_id)}")
    return " · ".join(details)
