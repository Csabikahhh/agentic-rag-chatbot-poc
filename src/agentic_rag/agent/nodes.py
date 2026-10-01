"""Node functions of the main agentic workflow (plan section 5.1).

The main graph has seven nodes, so the assignment's minimum of five still holds if two of them
are merged later. They share :class:`~agentic_rag.agent.state.AgentState`:

- ``analyze_request`` (LLM): classifies the latest user message and extracts the question.
- ``plan_subtasks`` (LLM): decomposes a complex request into independent sub-tasks, and
  re-plans after a rejected draft.
- ``run_rag_subtask`` (subgraph call): answers one ``retrieve`` sub-task with the RAG
  subgraph, through the ``search_knowledge_base`` tool.
- ``call_tool`` (action): executes one ``tool`` sub-task with a non-retrieval tool.
- ``synthesize_answer`` (LLM): writes a draft answer from the sub-task results.
- ``verify_answer`` (LLM): checks that the draft is grounded in the sub-task results.
- ``finalize_response`` (data): turns the draft into the final answer, its sources and the
  assistant message.

``run_rag_subtask`` and ``call_tool`` are the ``Send`` workers: each call receives one
:class:`~agentic_rag.agent.state.SubtaskInput` instead of the shared state, and LangGraph runs
the workers of a planning round in parallel (see ``agentic_rag.agent.routing``).

Conventions for every node (the node convention and the execution model are described once,
in ``agentic_rag.agent.graph``):

- It is a sync function named after its node and decorated with
  ``agentic_rag.tracing.traced``, so each execution appends one ``TraceEvent`` with the node's
  name to ``trace``.
- The state (or the ``Send`` payload) is the only positional parameter. Dependencies are
  keyword-only parameters without defaults, bound by ``build_agent_graph``: ``chat_model``
  (from ``agentic_rag.llm.get_chat_model``), ``tools`` (the non-retrieval tools by name) and
  ``search_tool`` (the ``search_knowledge_base`` tool). Tests call a node directly with a
  scripted chat model or a fake tool.
- It returns a ``dict`` partial update and never mutates the state. Keys without a reducer may
  be missing, so they are read with ``state.get(...)``.
- LLM nodes use structured output into Pydantic models (plan decision 7) and format their
  prompts from the raw state inside the node.
- Failures: transient errors (for example an unreachable Ollama server) are retried by the
  node's ``RetryPolicy``; a failing tool becomes a ``SubtaskResult`` with ``ok=False``, so the
  answer can still be written from the other results; unexpected errors propagate.

Citations: the ``[n]`` markers of a RAG context, and so of a ``SubtaskResult.output``, are
local to one RAG run. They number that result's own ``sources``, so every ``retrieve``
sub-task starts again at ``[1]``. ``synthesize_answer``, the fan-in point, therefore assigns
one global numbering: the ``sources`` of ``subtask_results`` in sub-task order, each result's
in rank order, de-duplicated by ``chunk_id`` (a repeated chunk keeps its first number). It
builds its prompt from these numbered sources, not from the per-run markers, and
``finalize_response`` returns exactly this list as ``sources``. The answer's ``[n]`` is then
``sources[n - 1]``, the number the UI shows. Both nodes derive the list from
``subtask_results`` with one shared function, so their numbers agree.

Skeleton: every node raises ``agentic_rag.errors.PlannedFeatureError`` until Phase 4 (plan
section 8).
"""

from collections.abc import Mapping
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from agentic_rag.agent.state import AgentState, SubtaskInput
from agentic_rag.agent.tools import SearchKnowledgeBaseTool
from agentic_rag.errors import planned
from agentic_rag.tracing import traced

__all__ = [
    "analyze_request",
    "call_tool",
    "finalize_response",
    "plan_subtasks",
    "run_rag_subtask",
    "synthesize_answer",
    "verify_answer",
]


@traced
def analyze_request(
    state: AgentState, *, chat_model: BaseChatModel, tools: Mapping[str, BaseTool]
) -> dict[str, Any]:
    """Classify the latest user message and extract the question.

    Kind: LLM (structured output). The first node of every run.

    Reads ``messages``: the latest user message, with the earlier turns as context.

    Writes ``question`` (a standalone question: references to earlier turns are resolved) and
    ``intent`` on every route; ``AgentOutput`` requires both, and ``route_after_analyze`` routes
    by ``intent``. Depending on the intent it also writes:

    - ``direct``: ``draft_answer``, the reply to a request that needs neither retrieval nor a
      tool (a greeting, a question about the assistant); ``finalize_response`` delivers it.
    - ``single``: ``subtasks`` with exactly one ``retrieve`` sub-task, whose input is a
      self-contained search query.
    - ``tool``: ``subtasks`` with exactly one ``tool`` sub-task, whose ``tool_name`` and
      ``tool_args`` come from the tool catalog; chosen only when a registered tool fits.
    - ``complex``: nothing more; ``plan_subtasks`` decomposes the request.

    Args:
        state: The graph state; only ``messages`` is guaranteed to be present.
        chat_model: The chat model, used with structured output.
        tools: The non-retrieval tools by name; their names, descriptions and argument schemas
            let the model recognize the ``tool`` intent and fill in the sub-task.

    Returns:
        The partial state update described above.

    Raises:
        PlannedFeatureError: Always, until Phase 4.
    """
    raise planned(f"{__name__}.analyze_request", 4)


@traced
def plan_subtasks(
    state: AgentState, *, chat_model: BaseChatModel, tools: Mapping[str, BaseTool]
) -> dict[str, Any]:
    """Decompose the question into independent sub-tasks, or re-plan after verification.

    Kind: LLM (structured output: a list of ``Subtask`` records, plan decision 7). It runs for
    the ``complex`` intent, and again whenever ``route_after_verify`` sends a rejected draft
    back for a re-plan.

    Reads ``question``. On a re-plan it also reads ``verdict``, ``draft_answer`` and the
    previous round's ``subtask_results``, to target what the rejected draft was missing.

    Writes:

    - ``subtasks``: the new plan, which replaces the previous one. Every sub-task can run on its
      own: a ``retrieve`` sub-task carries a self-contained query, a ``tool`` sub-task the name
      of one of ``tools`` and its arguments. Ids are unique within the plan. The plan is never
      empty, because an empty fan-out would end the run without an answer (an empty model
      reply falls back to one ``retrieve`` sub-task for the question), and its size is capped,
      so the fan-out stays bounded.
    - ``subtask_results``: ``langgraph.types.Overwrite(kept)``, which resets the append reducer
      at the start of every planning round. ``kept`` is usually ``[]``; results of the previous
      round that are still trusted may be kept, and their sub-tasks are then not planned
      again. The workers of the new round append to the reset list.

    Args:
        state: The graph state, with ``question`` set by ``analyze_request``.
        chat_model: The chat model, used with structured output.
        tools: The non-retrieval tools by name: the catalog a ``tool`` sub-task can name.

    Returns:
        The partial state update described above.

    Raises:
        PlannedFeatureError: Always, until Phase 4.
    """
    raise planned(f"{__name__}.plan_subtasks", 4)


@traced
def run_rag_subtask(state: SubtaskInput, *, search_tool: SearchKnowledgeBaseTool) -> dict[str, Any]:
    """Answer one ``retrieve`` sub-task from the knowledge base.

    Kind: subgraph call. A ``Send`` worker: LangGraph runs it once per ``retrieve`` sub-task,
    in parallel with the other workers of the round.

    Reads the ``Send`` payload: ``subtask`` (its ``input`` is the search query) and
    ``question`` (the user's question, as context).

    It runs the RAG subgraph through the ``search_knowledge_base`` tool, invoked with a tool
    call so that the resulting ``ToolMessage`` carries the whole ``RagOutput`` as its artifact.
    The subgraph keeps its own state schema (``RagInput`` in, ``RagOutput`` out) and does not
    count towards the main graph's nodes.

    Writes:

    - ``subtask_results``: one ``SubtaskResult`` with ``kind="retrieve"``, the subgraph's
      ``context`` as ``output`` and its ``sources`` in rank order, appended by the reducer
      (fan-in). The context's markers number these sources only (see Citations in the module
      docstring). A tool error gives ``ok=False`` with a short ``error`` instead of failing the
      whole run.
    - ``trace``: the subgraph's own events (``RagOutput["trace"]``), forwarded so that the
      run's trace also holds the per-node latency of the RAG subgraph; ``traced`` appends this
      node's event after them.

    Args:
        state: The ``Send`` payload of one ``retrieve`` sub-task.
        search_tool: The ``search_knowledge_base`` tool, bound to the compiled RAG subgraph.

    Returns:
        The partial state update described above.

    Raises:
        PlannedFeatureError: Always, until Phase 4.
    """
    raise planned(f"{__name__}.run_rag_subtask", 4)


@traced
def call_tool(state: SubtaskInput, *, tools: Mapping[str, BaseTool]) -> dict[str, Any]:
    """Execute one ``tool`` sub-task with a non-retrieval tool.

    Kind: action (no LLM call). A ``Send`` worker: LangGraph runs it once per ``tool``
    sub-task, in parallel with the other workers of the round.

    Reads the ``Send`` payload: ``subtask.tool_name`` selects the tool, ``subtask.tool_args``
    are its arguments, and ``question`` is context only.

    Writes ``subtask_results``: one ``SubtaskResult`` with ``kind="tool"`` and the tool's output
    as text, appended by the reducer (fan-in). An unknown tool name, arguments that fail the
    tool's schema, or a ``ToolException`` give ``ok=False`` with a short ``error``, so the answer
    can still be written from the other results and ``verify_answer`` sees the gap.

    Args:
        state: The ``Send`` payload of one ``tool`` sub-task.
        tools: The non-retrieval tools by name.

    Returns:
        The partial state update described above.

    Raises:
        PlannedFeatureError: Always, until Phase 4.
    """
    raise planned(f"{__name__}.call_tool", 4)


@traced
def synthesize_answer(state: AgentState, *, chat_model: BaseChatModel) -> dict[str, Any]:
    """Write a draft answer from the results of the current planning round.

    Kind: LLM. It runs once per round, after every ``Send`` worker of the round has finished
    (fan-in through the ``subtask_results`` reducer).

    Reads ``question``, ``subtask_results`` (sources, tool outputs and failures of the current
    round) and ``messages`` (the conversation, as context). The prompt presents the retrieved
    chunks under the global numbers of the citation contract (see Citations in the module
    docstring) instead of the per-run markers of ``SubtaskResult.output``; a tool result
    contributes its ``output``.

    Writes ``draft_answer``: an answer grounded only in the sub-task results that cites the
    retrieved chunks by those global ``[n]`` numbers; what the results do not cover is stated
    as unknown rather than guessed.

    Args:
        state: The graph state after the fan-in.
        chat_model: The chat model.

    Returns:
        The partial state update described above.

    Raises:
        PlannedFeatureError: Always, until Phase 4.
    """
    raise planned(f"{__name__}.synthesize_answer", 4)


@traced
def verify_answer(state: AgentState, *, chat_model: BaseChatModel) -> dict[str, Any]:
    """Check that the draft answer is supported by the sub-task results.

    Kind: LLM (structured output). Its verdict drives the bounded re-plan loop
    (``route_after_verify``).

    Reads ``question``, ``draft_answer`` and ``subtask_results``, plus ``verdict`` and
    ``retry_count`` of the previous verification, if there was one.

    Writes:

    - ``verdict``: ``"grounded"`` when the draft answers the question and every claim in it is
      supported by the results; ``"insufficient"`` otherwise.
    - ``retry_count``: the number of re-plans performed so far. It is 0 when the first draft is
      verified and grows by one for each draft that comes from a re-plan (that is, when the
      previous verdict was ``"insufficient"``). ``route_after_verify`` compares it with
      ``Settings.max_retries``.

    Args:
        state: The graph state, with ``draft_answer`` set by ``synthesize_answer``.
        chat_model: The chat model, used with structured output.

    Returns:
        The partial state update described above.

    Raises:
        PlannedFeatureError: Always, until Phase 4.
    """
    raise planned(f"{__name__}.verify_answer", 4)


@traced
def finalize_response(state: AgentState) -> dict[str, Any]:
    """Turn the draft into the final answer, its sources and the assistant message.

    Kind: data (no LLM call). The last node of every run, reached on every route.

    Reads ``draft_answer`` (written by ``analyze_request`` on the ``direct`` route and by
    ``synthesize_answer`` otherwise), ``verdict`` and ``subtask_results``.

    Writes, on every route (``AgentOutput`` requires ``answer`` and ``sources``):

    - ``answer``: the draft. When the last verdict is still ``"insufficient"`` (the retries
      are exhausted), it carries an explicit note that the question is only partially
      answered.
    - ``sources``: exactly the globally numbered list that ``synthesize_answer`` cited from
      (see Citations in the module docstring), so ``[n]`` in the answer is ``sources[n - 1]``;
      ``[]`` on the ``direct`` route. If it keeps only the chunks the answer cites, it
      renumbers the markers in ``answer`` to match the shorter list.
    - ``messages``: the assistant message with the answer, appended by ``add_messages``.

    ``traced`` appends the final event of the run's trace.

    Args:
        state: The graph state.

    Returns:
        The partial state update described above.

    Raises:
        PlannedFeatureError: Always, until Phase 4.
    """
    raise planned(f"{__name__}.finalize_response", 4)
