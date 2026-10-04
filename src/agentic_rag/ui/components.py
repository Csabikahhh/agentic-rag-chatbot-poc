"""Building blocks of the Streamlit chat UI.

The entrypoint (``app.py``) keeps the conversation in ``st.session_state`` as a list of
:class:`ChatTurn` records and replays it on every rerun with :func:`render_turn`. Every
assistant turn has the same three parts, in this order (:func:`render_assistant_turn`):

1. :func:`render_trace`: the main steps the agent took (node name, duration, summary) as a
   collapsible step timeline. The steps that LangGraph ran in one step, the parallel ``Send``
   workers of one planning round, are grouped into one "in parallel" step
   (:func:`group_parallel_steps`). A worker that ran the RAG subgraph lists the subgraph's
   steps under its own (:class:`AgentStep`): the rewritten search query, the retrieved and the
   kept chunks.
2. :func:`render_reply`: the answer, a notice when the run reached a part of the agent that is
   planned for a later phase, a short notice when the user stopped the run, or the failure:
   what to do about it for the failures :func:`describe_failure` knows (a missing index, an
   unreachable Ollama), the exception with its traceback for any other.
3. :func:`render_sources`: the RAG result: every retrieved chunk with its source, title, page,
   section, score and text, collapsed by default.

:func:`render_settings` shows the effective configuration in the sidebar, and
:data:`EXAMPLE_QUESTIONS` are the questions the empty chat offers.

Two helpers keep the history consistent for the agent: :func:`add_reply` puts an assistant
turn right after the question it replies to, and :func:`agent_messages` builds the
conversation the agent receives from the answered questions and the new one.

:class:`ChatTurn` lives in this module and not in the script because Streamlit executes the
script again on every rerun: a class defined there would be a new class on each run, while the
records in the session state would still be instances of the old one.

Text that comes from data (node names, summaries, titles, error messages) is escaped before it
is rendered as Markdown, and chunk text is shown verbatim with ``st.text``. Only the answer is
rendered as Markdown, because the model writes Markdown on purpose; its dollar signs are
escaped first (:func:`escape_dollar_signs`), so that currency amounts are not read as LaTeX.
"""

import json
import re
import sys
from collections.abc import Mapping, MutableSequence, Sequence
from typing import Any, Final, Literal
from urllib.parse import quote, urlsplit

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, ConfigDict, Field
from streamlit.delta_generator import DeltaGenerator

from agentic_rag import __version__
from agentic_rag.config import Settings
from agentic_rag.errors import EmbeddingMismatchError, IndexNotFoundError
from agentic_rag.rag.state import Source
from agentic_rag.tracing import TraceEvent, trace_events_from_chunk

__all__ = [
    "EXAMPLE_QUESTIONS",
    "AgentStep",
    "ChatTurn",
    "FailureHint",
    "Role",
    "TraceState",
    "add_reply",
    "agent_messages",
    "agent_steps_from_chunk",
    "describe_failure",
    "escape_dollar_signs",
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

EXAMPLE_QUESTIONS: Final = (
    "Which CSS pseudo-class selects a parent element that contains a specific child?",
    "Mi a különbség a React `useState` és a Nuxt `useState` között?",
    "Is `#777777` text on a white background readable enough?",
    "Which selector has the higher specificity: `#nav a` or `.menu li a`?",
    "Is `:has()` supported in Safari 15?",
)
"""Questions the empty chat offers, one per route of the agent: a single search, a comparison
that runs two searches in parallel (asked in Hungarian), and one question per tool (contrast,
specificity, browser support). The fake LLM provider routes them the same way."""

Role = Literal["user", "assistant"]
"""Author of a chat turn; also the name passed to ``st.chat_message``."""

TraceState = Literal["running", "complete", "stopped", "error"]
"""State of the step panel: the run is in progress, has finished, was stopped or has failed."""

# Icon of the step panel per state. The panel is an `st.expander`, not an `st.status`: in
# Streamlit 1.64 every `st.status` sleeps 50 ms when it is created, which would delay every
# live update of the panel and every assistant turn replayed on a rerun.
_PANEL_ICONS: dict[TraceState, str] = {
    "running": "spinner",
    "complete": ":material/account_tree:",
    "stopped": ":material/stop_circle:",
    "error": ":material/error:",
}

# Label prefix of the step panel of a finished run, per state.
_PANEL_TITLES: dict[TraceState, str] = {
    "complete": "Agent steps",
    "stopped": "Agent steps before the stop",
    "error": "Agent steps before the error",
}

# Characters that start inline Markdown (emphasis, code, links, HTML, tables, strikethrough,
# LaTeX) or a heading. Other line-start syntax (list markers, quotes) is left alone: labels
# render inline only, and in a body it can at most turn a line into a list item or a quote.
_MARKDOWN_SPECIALS = re.compile(r"([\\`*_\[\]<>#|~$])")
_BACKTICK_RUN = re.compile(r"`+")

# The block syntax that escape_dollar_signs recognizes, matched at the start of one line.
_FENCE_OPEN = re.compile(r"[ \t]*(?P<fence>`{3,}|~{3,})(?P<info>.*)")
_LIST_ITEM = re.compile(r"[ \t]*(?:[-+*]|\d{1,9}[.)])(?:[ \t]|$)")
_ATX_HEADING = re.compile(r" {0,3}#{1,6}(?:[ \t]|$)")
# The inline tokens that matter for dollar signs: a backslash escape, a backtick run (which
# may open a code span) and a dollar sign.
_INLINE_TOKEN = re.compile(r"\\.|`+|\$", re.DOTALL)


class AgentStep(BaseModel):
    """One main step of an agent run, with the subgraph steps it ran.

    Attributes:
        event: The trace event that a node of the main graph recorded itself.
        substeps: The events that the node forwarded from a subgraph, in execution order:
            the RAG subgraph's steps for ``run_rag_subtask``, empty for the other nodes.
    """

    model_config = ConfigDict(frozen=True)

    event: TraceEvent
    substeps: list[TraceEvent] = Field(default_factory=list)


class FailureHint(BaseModel):
    """What a failure of the run means for the user, and what to do about it.

    Attributes:
        title: A short title, such as "Ollama is not reachable".
        message: The explanation and the remedy, as Markdown.
    """

    model_config = ConfigDict(frozen=True)

    title: str
    message: str


class ChatTurn(BaseModel):
    """One message of the conversation, as kept in ``st.session_state``.

    A user turn holds the question in ``content``. An assistant turn holds the outcome of one
    agent run: the answer in ``content`` (Markdown), the main steps in ``trace`` and the
    retrieved chunks in ``sources``. A run that ends without an answer marks the turn
    instead: ``notice`` when the run reached a part of the agent that is planned for a later
    phase, ``stopped`` when the user stopped it, ``error`` when it failed (with ``hint`` when
    :func:`describe_failure` knows the failure). ``trace`` then holds the steps that finished
    before. Turns are immutable values.

    Attributes:
        role: Author of the turn.
        content: The question, or the answer in Markdown; empty when there is no answer.
        trace: The main steps of the run, in the order they finished.
        sources: The retrieved chunks behind the answer, in citation order.
        notice: Message of the ``PlannedFeatureError`` that ended the run.
        stopped: Whether the user stopped the run before it produced an answer.
        error: The exception that made the run fail.
        hint: What ``error`` means and how to fix it, for a known failure; shown instead of
            the traceback.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    role: Role
    content: str = ""
    trace: list[AgentStep] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    notice: str | None = None
    stopped: bool = False
    error: Exception | None = None
    hint: FailureHint | None = None

    @property
    def answered(self) -> bool:
        """Whether this is an assistant turn with an answer that the model wrote."""
        return (
            self.role == "assistant"
            and bool(self.content)
            and self.notice is None
            and not self.stopped
            and self.error is None
        )

    def as_message(self) -> BaseMessage | None:
        """Return the turn as a message of the conversation history the agent receives.

        Returns:
            A ``HumanMessage`` for a user turn, an ``AIMessage`` for an assistant turn with an
            answer, and None for an assistant turn without one (a notice, a stop or an error):
            the model did not write those, so they are not part of its history.
        """
        if self.role == "user":
            return HumanMessage(content=self.content)
        if self.answered:
            return AIMessage(content=self.content)
        return None


def agent_messages(history: Sequence[ChatTurn]) -> list[BaseMessage]:
    """Build the conversation that the agent receives for the last question of a history.

    The last turn is the new question and is always kept. An earlier question is kept only
    when an assistant turn with an answer follows it, and then together with that answer. A
    question whose run ended with a notice, a stop or an error is left out with its reply,
    and so is a question that a stopped run has not closed yet (see :func:`add_reply`). The
    agent therefore receives user and assistant messages in turn, ending with the new
    question, and never a question that went unanswered.

    Args:
        history: The conversation, oldest first; its last turn is the new question.

    Returns:
        ``HumanMessage`` and ``AIMessage`` objects, oldest first.
    """
    # A snapshot: a stopped run that winds down in another thread may still add its reply.
    turns = tuple(history)
    messages: list[BaseMessage] = []
    for index, turn in enumerate(turns):
        message = turn.as_message()
        if message is None:
            continue
        is_new_question = index == len(turns) - 1
        if turn.role == "user" and not is_new_question and not turns[index + 1].answered:
            continue
        messages.append(message)
    return messages


def add_reply(history: MutableSequence[ChatTurn], question: ChatTurn, reply: ChatTurn) -> None:
    """Insert an assistant turn into the history, right after the question it replies to.

    The question is normally the last turn, so the reply is appended. With Streamlit's
    ``runner.fastReruns`` (on by default), though, a rerun stops the running script and starts
    the next run at once, while the stopped one winds down in its own thread; its reply can
    then arrive after the user has asked a newer question. Inserting it next to its own
    question keeps every question directly followed by its reply.

    The function only changes the list and calls no Streamlit command, so it is safe in the
    ``finally`` block of a run that the user stopped.

    Args:
        history: The conversation, oldest first; changed in place.
        question: The user turn that ``reply`` answers, as stored in ``history``. It is found
            by identity, so an earlier question with the same text does not get the reply.
        reply: The assistant turn to insert.
    """
    position = next(
        (index + 1 for index, turn in enumerate(history) if turn is question), len(history)
    )
    history.insert(position, reply)


def agent_steps_from_chunk(part: object) -> list[AgentStep]:
    """Extract the main steps from one ``"updates"`` part of a ``version="v2"`` graph stream.

    Each node's update carries the event the node recorded itself, after the events it
    forwarded from a subgraph (see ``agentic_rag.tracing``). The node's own event becomes an
    :class:`AgentStep` and the forwarded events its ``substeps``. Parts of other types, and
    updates without an event of their own node, yield no steps.

    Args:
        part: One item of ``graph.stream(..., stream_mode=[...], version="v2")``.

    Returns:
        The main steps of the part, in the order of its updates.
    """
    if not isinstance(part, Mapping) or part.get("type") != "updates":
        return []
    data = part.get("data")
    if not isinstance(data, Mapping):
        return []
    steps: list[AgentStep] = []
    for node, update in data.items():
        events = trace_events_from_chunk({node: update})
        forwarded = [event for event in events if event.node != node]
        steps.extend(
            AgentStep(event=event, substeps=forwarded) for event in events if event.node == node
        )
    return steps


def describe_failure(error: BaseException, settings: Settings) -> FailureHint | None:
    """Explain a failure that the user can fix, such as a missing index or a stopped Ollama.

    Known failures, checked on the exception and on the exceptions it was raised from:

    - ``IndexNotFoundError``: the knowledge base has not been built yet;
    - ``EmbeddingMismatchError``: the index was built with other embeddings;
    - with the ``ollama`` LLM provider, a connection to ``OLLAMA_BASE_URL`` that failed or
      timed out (``httpx.ConnectError``, ``httpx.ConnectTimeout``), and a model that Ollama
      does not have (``ollama.ResponseError`` with status 404).

    The function imports nothing: the HTTP and Ollama clients can only have raised when they
    are loaded, so it looks them up in ``sys.modules``.

    Args:
        error: The exception that made the run fail.
        settings: The effective settings, for the Ollama URL and model in the remedy.

    Returns:
        The explanation, or None for any other failure, which keeps its traceback.
    """
    for cause in _causes(error):
        if isinstance(cause, IndexNotFoundError):
            return FailureHint(
                title="The knowledge base has not been built yet",
                message=(
                    "The agent cannot search the documentation, because the vector index is "
                    "missing. Build it with `agentic-rag ingest --download`, which also "
                    "downloads the corpus, then ask again.\n\n"
                    f"{escape_markdown(str(cause))}"
                ),
            )
        if isinstance(cause, EmbeddingMismatchError):
            return FailureHint(
                title="The index does not match the embedding settings",
                message=(
                    f"{escape_markdown(str(cause))}\n\n"
                    "After a rebuild, ask again; after a change of the settings, restart the app."
                ),
            )
        if settings.llm_provider == "ollama":
            hint = _describe_ollama_failure(cause, settings)
            if hint is not None:
                return hint
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
    render_trace(turn.trace, state=_trace_state(turn))
    render_reply(turn)
    render_sources(turn.sources)


def render_reply(turn: ChatTurn) -> None:
    """Render the reply of an assistant turn.

    By priority: the failure of a run (its ``hint`` as an error message when the failure is a
    known one, otherwise the exception with its traceback, ``st.exception``), the notice of a
    part of the agent that is not built yet, the notice of a run that the user stopped, the
    answer as Markdown (with its dollar signs escaped, see :func:`escape_dollar_signs`), or a
    warning when the run finished without an answer.

    Args:
        turn: An assistant turn.
    """
    if turn.error is not None and turn.hint is not None:
        st.error(turn.hint.message, title=turn.hint.title, icon=":material/error:")
    elif turn.error is not None:
        st.exception(turn.error)
    elif turn.notice is not None:
        st.info(
            f"The agent cannot answer yet: {escape_markdown(turn.notice)}\n\n"
            "The chat interface works and the conversation is kept.",
            title="Not available yet",
            icon=":material/construction:",
        )
    elif turn.stopped:
        st.info(
            "Stopped before an answer was produced. Ask again to get one.",
            icon=":material/stop_circle:",
        )
    elif turn.content:
        st.markdown(link_citations(escape_dollar_signs(turn.content), turn.sources))
    else:
        st.warning("The agent finished without an answer.", icon=":material/help:")


def render_trace(steps: Sequence[AgentStep], *, state: TraceState = "complete") -> None:
    """Render the main steps of an agent run as a collapsible step timeline.

    The panel is a compact expander whose label sums up the run (the number of main steps and
    the wall-clock time) and whose icon shows the state. Inside, the steps follow each other
    in the order they started, each with its node name and duration in the label and its
    summary, start offset and metadata in the body. The steps of one LangGraph step form one
    "in parallel" step with a table of them (:func:`group_parallel_steps`). The subgraph steps
    of a main step are listed in a table under it, or as indented rows under it in the table
    of a parallel step.

    The live panel is redrawn in place, and a redraw keeps the elements of the one before that
    it does not overwrite. Two rules keep it from leaving any behind: every timeline step has
    the same three elements (summary, details, table; empty placeholders where there is
    nothing to show), and the timeline only grows, because a new step either starts a group
    or joins the last one.

    Args:
        steps: Main steps in any order, as :func:`agent_steps_from_chunk` returns them; may
            be empty.
        state: ``"running"`` while the run is in progress: the panel is open, with a spinner
            and an animated label. ``"complete"`` after a run, ``"stopped"`` after a run that
            the user stopped and ``"error"`` after a failed run: the panel is collapsed, or
            replaced by a short caption when there are no steps.
    """
    if not steps and state != "running":
        st.caption(":material/account_tree: No agent steps were recorded for this turn.")
        return
    events = [step.event for step in steps]
    panel = st.expander(
        _trace_label(events, state),
        expanded=state == "running",
        icon=_PANEL_ICONS[state],
        type="compact",
    )
    origin = min(event.started_at for event in events) if events else 0.0
    for group in group_parallel_steps(steps):
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
    st.table(settings_summary(settings), border="horizontal", width="stretch")
    method = "Hybrid (vector + keyword)" if settings.hybrid_search else "Vector only"
    candidates = max(settings.top_k, settings.retrieval_candidates)
    st.caption(f"{method} · {candidates} candidates → up to {settings.top_k} context chunks")
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
        model, the retrieval depth (top-k) and whether the LLM grades the retrieved chunks
        (``GRADE_WITH_LLM``, which the fake LLM provider never does).
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
    if settings.llm_provider == "fake":
        grading = "off · not with the fake LLM"
    else:
        grading = "on" if settings.grade_with_llm else "off · score threshold only"
    summary[":material/rule: LLM grading"] = grading
    return summary


def group_parallel_steps(steps: Sequence[AgentStep]) -> list[list[AgentStep]]:
    """Order main steps by start time and group the ones that LangGraph ran in parallel.

    LangGraph runs a graph in steps (supersteps): the nodes of one step run in parallel, the
    ``Send`` workers of one planning round for example, and the next step starts when they
    have all finished. Every event records its step (``TraceEvent.step``), so the steps that
    share one form a group, also when a fast worker, such as a pure-Python tool, finished
    before the next one started. Each sequential step of the main workflow is a group of its
    own.

    An event without a step (recorded outside a graph run) is grouped by time instead: with
    the group before it when it starts before that group ends. Events that only touch (one
    starts exactly when the other ends) are sequential.

    Args:
        steps: Main steps in any order.

    Returns:
        The groups in chronological order. Inside a group, the steps are ordered by start
        time, then by end time.
    """
    groups: list[list[AgentStep]] = []
    group_end = 0.0
    for step in sorted(steps, key=lambda item: (item.event.started_at, item.event.ended_at)):
        event = step.event
        if groups and _same_round(groups[-1][-1].event, event, group_end):
            groups[-1].append(step)
            group_end = max(group_end, event.ended_at)
        else:
            groups.append([step])
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


def escape_dollar_signs(markdown: str) -> str:
    """Escape the dollar signs that Streamlit's Markdown would read as LaTeX math.

    ``st.markdown`` renders the text between two dollar signs as inline math, so an answer
    such as "The fee is $25 and the late fee is $40" would lose both amounts. This function
    puts a backslash in front of every dollar sign of the prose, which Markdown then shows as
    a plain ``$``; the rest of the Markdown is unchanged. It leaves alone the places where a
    backslash would show literally, and dollar signs that are already escaped:

    - inline code spans, also when one runs across the lines of a paragraph;
    - fenced code blocks, with backtick or tilde fences, also indented ones as in list items;
      a fence that is never closed runs to the end of the text;
    - indented code blocks: lines indented by four or more columns that follow a blank line,
      a heading or a code block, outside lists.

    The block rules are a small subset of CommonMark that covers the Markdown models write.
    Inside a list, an indented line counts as prose; an indented code block nested in a list
    item would show its escaped dollar signs with the backslash. Math that the model meant as
    math shows as its source text.

    Args:
        markdown: Markdown text, typically an answer of the model.

    Returns:
        The text with ``\\$`` for every unescaped dollar sign of the prose.
    """
    if "$" not in markdown:
        return markdown
    classifier = _LineClassifier()
    output: list[str] = []
    prose: list[str] = []  # The lines of the current prose block.

    def flush_prose() -> None:
        if prose:
            output.append(_escape_inline_dollars("\n".join(prose)))
            prose.clear()

    for line in markdown.split("\n"):
        kind = classifier.classify(line)
        if kind != "prose":
            flush_prose()  # Any other line ends the prose block: code spans stay inside one.
        if kind in ("prose", "item"):
            prose.append(line)
        elif kind == "heading":
            output.append(_escape_inline_dollars(line))
        else:
            output.append(line)  # Code or a blank line, as it is.
    flush_prose()
    return "\n".join(output)


type _LineKind = Literal["blank", "code", "heading", "item", "prose"]
"""What a line of Markdown is to :func:`escape_dollar_signs`.

``code`` is a line of a fenced or indented code block, fences included; ``item`` is the first
line of a list item; ``prose`` is any other line of a paragraph or list item.
"""


class _LineClassifier:
    """Tells, line by line, which lines of a Markdown text are code and which are prose.

    Feed it every line in order: whether a line is code depends on the lines before it.
    """

    def __init__(self) -> None:
        self._fence: str | None = None  # The opening fence of the open fenced code block.
        self._in_paragraph = False  # The previous line was prose that a next line continues.
        self._in_list = False

    def classify(self, line: str) -> _LineKind:
        """Return what the next line of the text is, and remember it for the lines after it."""
        if self._fence is not None:
            if _closes_fence(line, self._fence):
                self._fence = None
            return "code"
        if not line.strip():
            self._in_paragraph = False
            return "blank"
        indent = _indent_width(line)
        if indent >= 4 and not self._in_paragraph and not self._in_list:
            return "code"  # A line of an indented code block.
        is_item = _LIST_ITEM.match(line) is not None
        # A list goes on until an unindented line starts a new block after it.
        self._in_list = is_item or (self._in_list and (indent > 0 or self._in_paragraph))
        if self._opens_fence(line, indent):
            return "code"
        if _ATX_HEADING.match(line):
            self._in_paragraph = False
            return "heading"
        self._in_paragraph = True
        return "item" if is_item else "prose"

    def _opens_fence(self, line: str, indent: int) -> bool:
        """Open a fenced code block if the line is an opening fence; return whether it is."""
        opening = _FENCE_OPEN.match(line)
        if opening is None or (indent > 3 and not self._in_list):
            return False
        fence = opening["fence"]
        if fence.startswith("`") and "`" in opening["info"]:
            return False  # Inline code on one line, such as ```x```, not a fence.
        self._fence = fence
        self._in_paragraph = False
        return True


def _escape_inline_dollars(text: str) -> str:
    """Escape the unescaped dollar signs of inline Markdown, outside its code spans."""
    parts: list[str] = []
    position = 0
    while (token := _INLINE_TOKEN.search(text, position)) is not None:
        start, end = token.span()
        parts.append(text[position:start])
        lexeme = token.group()
        if lexeme == "$":
            parts.append("\\$")
        else:
            if lexeme.startswith("`"):
                end = _code_span_end(text, end, len(lexeme))
            parts.append(text[start:end])  # A backslash escape or a code span, as it is.
        position = end
    parts.append(text[position:])
    return "".join(parts)


def _code_span_end(text: str, start: int, length: int) -> int:
    """Return where a code span ends whose opening backtick run ends at ``start``.

    The span closes with the next run of exactly ``length`` backticks. Without one, the
    opening run is literal text, and ``start`` is returned.
    """
    for run in _BACKTICK_RUN.finditer(text, start):
        if len(run.group()) == length:
            return run.end()
    return start


def _closes_fence(line: str, fence: str) -> bool:
    """Return whether a line closes the fenced code block that ``fence`` opened."""
    marker = line.strip()
    return len(marker) >= len(fence) and marker == fence[0] * len(marker)


def _indent_width(line: str) -> int:
    """Return the indentation of a line in columns, with a tab stop every four columns."""
    whitespace = line[: len(line) - len(line.lstrip(" \t"))]
    return len(whitespace.expandtabs(4))


def _causes(error: BaseException) -> list[BaseException]:
    """Return the exception and the exceptions it was raised from (``__cause__``), in order."""
    chain: list[BaseException] = []
    current: BaseException | None = error
    while current is not None and all(current is not seen for seen in chain):
        chain.append(current)
        current = current.__cause__
    return chain


def _describe_ollama_failure(error: BaseException, settings: Settings) -> FailureHint | None:
    """Explain an Ollama server that cannot be reached or lacks the model, else return None."""
    url, model = _inline_code(settings.ollama_base_url), _inline_code(settings.ollama_model)
    pull = _inline_code(f"ollama pull {settings.ollama_model}")
    httpx = sys.modules.get("httpx")
    if httpx is not None and isinstance(error, httpx.ConnectError | httpx.ConnectTimeout):
        try:
            target = error.request.url
        except RuntimeError:  # An error raised without a request.
            return None
        server = httpx.URL(settings.ollama_base_url)
        if (target.host, target.port) != (server.host, server.port):
            return None  # Another server, such as the Hugging Face Hub.
        return FailureHint(
            title="Ollama is not reachable",
            message=(
                f"The agent could not connect to Ollama at {url}. Start Ollama (`ollama serve`; "
                "in the container stack, `docker compose up`), or set `OLLAMA_BASE_URL` to "
                "where it runs and restart the app. Then ask again."
            ),
        )
    ollama = sys.modules.get("ollama")
    if ollama is not None and isinstance(error, ollama.ResponseError) and error.status_code == 404:
        return FailureHint(
            title="The model is not available",
            message=(
                f"Ollama at {url} does not have the model {model}. Pull it with {pull} (in "
                "the container stack, the `ollama-pull` service pulls it), or set "
                "`OLLAMA_MODEL` to a model that Ollama "
                "has and restart the app. Then ask again.\n\n"
                f"Ollama said: {escape_markdown(str(error.error))}"
            ),
        )
    return None


def _trace_state(turn: ChatTurn) -> TraceState:
    """Return the state of the step panel of a finished assistant turn."""
    if turn.error is not None:
        return "error"
    if turn.stopped:
        return "stopped"
    return "complete"


def _same_round(previous: TraceEvent, event: TraceEvent, group_end: float) -> bool:
    """Return whether an event belongs to the group of the event that started before it."""
    if previous.step is not None and event.step is not None:
        return event.step == previous.step
    return event.started_at < group_end


def _render_step(parent: DeltaGenerator, step: AgentStep, origin: float) -> None:
    """Render one step of the timeline: node and duration in the label, details in the body."""
    event = step.event
    body = parent.expander(
        f"{_inline_code(event.node)} · {format_duration(event.duration_ms)}",
        expanded=True,
        icon=":material/check_circle:",
        type="step",
    )
    details = [f"Started at +{_offset(event, origin)}", *_metadata_items(event.metadata)]
    entries = [(_inline_code(substep.node), substep) for substep in step.substeps]
    _fill_step(body, event.summary, details, _step_rows(entries, origin))


def _render_parallel_steps(
    parent: DeltaGenerator, group: Sequence[AgentStep], origin: float
) -> None:
    """Render the steps of one LangGraph step as one timeline step with a table of them."""
    entries: list[tuple[str, TraceEvent]] = []
    for step in group:
        entries.append((_inline_code(step.event.node), step.event))
        entries.extend((f"↳ {_inline_code(sub.node)}", sub) for sub in step.substeps)
    events = [step.event for step in group]
    body = parent.expander(
        f"{len(group)} steps in parallel · {format_duration(_wall_ms(events))}",
        expanded=True,
        icon=":material/call_split:",
        type="step",
    )
    details = [f"Started at +{_offset(events[0], origin)}"]
    if events[0].step is not None:
        details.append(f"LangGraph step {events[0].step}")
    _fill_step(body, "", details, _step_rows(entries, origin))


def _fill_step(
    body: DeltaGenerator, summary: str, details: Sequence[str], rows: Sequence[dict[str, str]]
) -> None:
    """Fill the body of a timeline step with its three elements, always all three.

    The summary, the details line and the table of the sub-steps or parallel steps, with an
    empty placeholder for a missing summary or table. The live panel is redrawn as the steps
    arrive, and a redrawn step keeps the elements of the earlier drawing that the new one does
    not overwrite: with three elements every time, a step that becomes a parallel step (when
    the second worker of its LangGraph step finishes) leaves nothing behind. The details line
    is never empty, because a step without content would end the timeline.
    """
    if summary:
        body.markdown(escape_markdown(summary))
    else:
        body.empty()
    body.caption(" · ".join(details))
    if rows:
        body.table(rows, hide_index=True, border="horizontal")
    else:
        body.empty()


def _step_rows(entries: Sequence[tuple[str, TraceEvent]], origin: float) -> list[dict[str, str]]:
    """Build the table rows of steps from (label, event) pairs; Details only when needed."""
    with_details = any(event.metadata for _, event in entries)
    rows: list[dict[str, str]] = []
    for label, event in entries:
        row = {
            "Step": label,
            "Start": f"+{_offset(event, origin)}",
            "Duration": format_duration(event.duration_ms),
            "Summary": escape_markdown(event.summary),
        }
        if with_details:
            row["Details"] = " · ".join(_metadata_items(event.metadata))
        rows.append(row)
    return rows


def _trace_label(events: Sequence[TraceEvent], state: TraceState) -> str:
    """Return the label of the step panel: what is running, or how many steps took how long."""
    if state == "running":
        progress = f" · {_count(len(events), 'step')} so far" if events else ""
        return f":shimmer[Running the agent]{progress}"
    summary = f"{_count(len(events), 'step')} · {format_duration(_wall_ms(events))}"
    return f"{_PANEL_TITLES[state]} · {summary}"


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
    """Format one metadata value: strings as they are, floats with four significant digits,
    anything else as compact JSON."""
    if isinstance(value, str):
        return value
    if isinstance(value, float):
        return f"{value:.4g}"
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
    if source.url and _is_web_url(source.url):
        details.append(f"[Open the page]({_link_target(source.url)})")
    if source.revision:
        details.append(f"Indexed snapshot: {_inline_code(source.revision[:12])}")
    if source.score is not None:
        details.append(f"Score: {source.score:.3f}")
    details.append(f"Chunk: {_inline_code(source.chunk_id)}")
    return " · ".join(details)


def _link_target(url: str) -> str:
    """Encode the characters that would end a Markdown link target early."""
    return quote(url, safe="/:#?=&%+@;,$!~*'-._")


def _is_web_url(url: str) -> bool:
    """Only make absolute HTTP(S) source URLs clickable."""
    try:
        parsed = urlsplit(url)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except ValueError:
        return False


def link_citations(markdown: str, sources: Sequence[Source]) -> str:
    """Resolve numbered citations using Markdown references, leaving code blocks untouched.

    Markdown's own parser distinguishes prose references from code and existing links.
    Sources without a public web URL retain their plain citation and context expander.
    """
    references = [
        f"[{number}]: <{_link_target(source.url)}>"
        for number, source in enumerate(sources, start=1)
        if source.url and _is_web_url(source.url)
    ]
    return markdown + "\n\n" + "\n".join(references) if references else markdown
