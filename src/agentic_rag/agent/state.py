"""State contracts of the main agentic workflow.

The seven nodes of the main graph (plan section 5.1) share :class:`AgentState`:

- ``analyze_request`` reads ``messages`` and writes ``question`` and ``intent``; on the
  ``direct`` route also ``draft_answer`` (the reply), and on the ``single`` and ``tool`` routes
  a one-step plan in ``subtasks``.
- ``plan_subtasks`` reads ``question`` (on a re-plan also ``verdict``, ``draft_answer`` and the
  previous ``subtask_results``) and writes ``subtasks``, resetting ``subtask_results`` (see
  below).
- ``run_rag_subtask`` and ``call_tool`` run once per sub-task: each receives a
  :class:`SubtaskInput` through ``Send`` and appends one :class:`SubtaskResult`.
- ``synthesize_answer`` reads ``question`` and ``subtask_results`` and writes ``draft_answer``.
- ``verify_answer`` reads ``draft_answer`` and ``subtask_results`` and writes ``verdict``,
  ``retry_count`` and ``critique`` (what a rejected draft is missing, for the re-plan).
- ``finalize_response`` writes ``answer``, ``sources`` and the assistant message.

The graph is compiled as ``StateGraph(AgentState, input_schema=AgentInput,
output_schema=AgentOutput)``: it starts from :class:`AgentInput` alone and returns the
:class:`AgentOutput` keys. Raw data lives in the state; prompts are formatted inside the nodes.
The literal types ``Intent``, ``Verdict`` and ``SubtaskKind`` live in
``agentic_rag.agent.types``, which does not import LangGraph; this module re-exports them.

Reads: ``messages``, ``subtask_results`` and ``trace`` have reducers and always exist (they
start as empty lists). Every other key is absent until a node writes it, so nodes read keys
that may be missing with ``state.get(...)``.

Re-planning: ``subtask_results`` uses a plain append reducer, so the results of the parallel
``Send`` workers fan in. Plain append alone would carry the results of a rejected round into
the next one after ``verify_answer`` returns ``"insufficient"``. Every planning round
therefore starts with a reset: ``plan_subtasks`` writes
``{"subtask_results": Overwrite(kept)}``, using LangGraph's ``langgraph.types.Overwrite``,
which bypasses the reducer. ``kept`` is usually ``[]``; the planner may keep results it still
trusts and then not plan those sub-tasks again. The workers of the new round append to the
reset list, so ``synthesize_answer`` sees only the current round (plus what was kept). Their
results arrive in plan order whatever order the workers finish in, because LangGraph applies
the writes of parallel ``Send`` tasks in ``Send`` order. ``subtasks`` has no reducer: each plan
replaces the previous one.

Trace: every node is wrapped with ``agentic_rag.tracing.traced``, which appends the node's
own event to ``trace``. ``run_rag_subtask`` also forwards ``RagOutput["trace"]`` (the events of
the RAG subgraph nodes) in its update, so the final trace covers the subgraph's per-node
latency as well.

Persistence: no checkpointer is used, so every request starts from a fresh state and the
conversation arrives in ``messages``. If one is added (for example for multi-turn memory), the
first node of each turn must reset every key except ``messages``, not only
``subtask_results`` and ``trace``; ``traced`` accepts only a list of events under ``trace``
today, so that reset needs support there first. Callers then send only the new message:
``add_messages`` appends messages that have no id, so a resent history would be stored twice.
"""

import operator
from typing import Annotated, Any, Required, Self, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentic_rag.agent.types import Intent, SubtaskKind, Verdict
from agentic_rag.rag.state import Source
from agentic_rag.tracing import TraceEvent

__all__ = [
    "AgentInput",
    "AgentOutput",
    "AgentState",
    "Intent",
    "Subtask",
    "SubtaskInput",
    "SubtaskKind",
    "SubtaskResult",
    "Verdict",
]


class Subtask(BaseModel):
    """One independently executable step of a plan.

    The planner emits a list of these as structured output; the field descriptions are part
    of the JSON schema the model sees. Sub-tasks are immutable values.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(
        min_length=1, description="Short identifier, unique within one plan, e.g. 's1'."
    )
    kind: SubtaskKind = Field(
        description="'retrieve' searches the knowledge base; 'tool' calls a non-retrieval tool."
    )
    input: str = Field(
        description=(
            "For 'retrieve': a self-contained search query. "
            "For 'tool': what the tool should compute, in plain words."
        )
    )
    tool_name: str | None = Field(
        default=None, description="Name of the tool to call; required when kind is 'tool'."
    )
    tool_args: dict[str, Any] = Field(
        default_factory=dict, description="Keyword arguments for the tool."
    )

    @model_validator(mode="after")
    def _check_tool_name(self) -> Self:
        """Require a tool name for tool sub-tasks, so they can be dispatched."""
        if self.kind == "tool" and not self.tool_name:
            raise ValueError("tool_name is required when kind is 'tool'")
        return self


class SubtaskResult(BaseModel):
    """Outcome of one sub-task, appended to ``AgentState.subtask_results``.

    ``ok`` is ``False`` exactly when ``error`` is set. Results are immutable values.
    """

    model_config = ConfigDict(frozen=True)

    subtask_id: str = Field(min_length=1, description="Id of the Subtask this result answers.")
    kind: SubtaskKind = Field(description="Kind of the sub-task.")
    output: str = Field(
        description=(
            "Text for synthesis: the RAG context, whose citation markers number this "
            "result's sources only, or the tool output; may be empty."
        )
    )
    sources: list[Source] = Field(
        default_factory=list, description="Chunks behind output, in rank order (retrieve)."
    )
    ok: bool = Field(default=True, description="False when the sub-task failed.")
    error: str | None = Field(default=None, description="Short, user-safe reason when ok is False.")

    @model_validator(mode="after")
    def _check_error(self) -> Self:
        """Keep ``ok`` and ``error`` consistent."""
        if self.ok == (self.error is not None):
            raise ValueError("error must be set exactly when ok is False")
        return self


class SubtaskInput(TypedDict):
    """Payload a ``Send`` delivers to ``run_rag_subtask`` and ``call_tool``.

    Attributes:
        subtask: The sub-task to execute.
        question: The user's question, as context for the sub-task.
    """

    subtask: Subtask
    question: str


class AgentInput(TypedDict):
    """Public input of the main graph.

    Attributes:
        messages: The conversation so far, ending with the user's new message. Plain
            ``("user", "...")`` tuples and dicts are accepted as well and become messages.
    """

    messages: list[AnyMessage]


class AgentOutput(TypedDict, total=False):
    """Public output of the main graph, returned by ``agent_graph.invoke``.

    The required keys are present after every successful run: ``analyze_request`` always
    writes ``question`` and ``intent``, ``finalize_response`` always writes ``answer`` and
    ``sources``, and the reducer keys always exist. The other keys depend on the route.

    Attributes:
        messages: The conversation including the assistant's final message.
        question: The question extracted from the last user message.
        intent: The route chosen by ``analyze_request``.
        answer: The final answer; it states explicitly when it is only partial.
        sources: The numbered sources of the answer: its marker ``[n]`` refers to
            ``sources[n - 1]`` (the citation contract in ``agentic_rag.agent.nodes``).
        subtask_results: Results of the last planning round, for the UI and the evaluation.
        trace: Every event of the run, including the forwarded RAG subgraph events.
        subtasks: The last plan; present when a plan was made.
        verdict: The last verification outcome; present when ``verify_answer`` ran.
        retry_count: Re-plans performed; present when ``verify_answer`` ran.
    """

    messages: Required[list[AnyMessage]]
    question: Required[str]
    intent: Required[Intent]
    answer: Required[str]
    sources: Required[list[Source]]
    subtask_results: Required[list[SubtaskResult]]
    trace: Required[list[TraceEvent]]
    subtasks: list[Subtask]
    verdict: Verdict | None
    retry_count: int


class AgentState(TypedDict, total=False):
    """Shared memory of the main graph nodes.

    Only ``messages`` is required, so the graph starts from :class:`AgentInput` alone. See the
    module docstring for the reducers, the re-plan reset and the trace forwarding.

    Attributes:
        messages: Conversation history; ``add_messages`` appends (and updates by message id).
        question: The question extracted from the last user message (``analyze_request``).
        intent: The route chosen by ``analyze_request``.
        subtasks: The current plan, written by ``plan_subtasks`` (a one-step plan by
            ``analyze_request`` on the ``single`` and ``tool`` routes); each plan replaces the
            previous one.
        subtask_results: Results of the current planning round; appended by the ``Send``
            workers and reset with ``Overwrite`` by ``plan_subtasks``.
        draft_answer: Answer candidate written by ``synthesize_answer`` (by ``analyze_request``
            on the ``direct`` route).
        verdict: Outcome of ``verify_answer``; ``None`` or absent before verification.
        critique: What ``verify_answer`` found missing or unsupported in the draft; empty
            when the draft is grounded. ``plan_subtasks`` targets it on a re-plan, and
            ``finalize_response`` names it when the answer stays partial.
        language: The language of the latest user message, named in English (``Hungarian``),
            written by ``analyze_request``; ``synthesize_answer`` answers in it.
        retry_count: Number of re-plans so far, written by ``verify_answer`` and bounded by
            ``Settings.max_retries``.
        answer: Final answer written by ``finalize_response``.
        sources: The numbered sources of the answer, written by ``finalize_response``.
        trace: Trace events of the run, appended by ``traced``.
    """

    messages: Required[Annotated[list[AnyMessage], add_messages]]
    question: str
    intent: Intent
    subtasks: list[Subtask]
    subtask_results: Annotated[list[SubtaskResult], operator.add]
    draft_answer: str
    verdict: Verdict | None
    critique: str
    language: str
    retry_count: int
    answer: str
    sources: list[Source]
    trace: Annotated[list[TraceEvent], operator.add]
