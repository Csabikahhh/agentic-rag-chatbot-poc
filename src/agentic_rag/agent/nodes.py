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

Every node follows the node convention of ``agentic_rag.agent.graph`` (the state as the only
positional parameter, keyword-only dependencies bound by ``build_agent_graph``, a ``dict``
partial update), and in addition:

- It is named after its node and decorated with ``agentic_rag.tracing.traced``, so each
  execution appends one ``TraceEvent`` with the node's name and a one-line summary to
  ``trace``.
- Its dependencies are ``chat_model`` (from ``agentic_rag.llm.get_chat_model``), ``tools``
  (the non-retrieval tools by name) and ``search_tool`` (the ``search_knowledge_base`` tool).
  Tests call a node directly with a scripted chat model or a fake tool.
- It never mutates the state, and reads the keys without a reducer, which may be missing,
  with ``state.get(...)``.
- LLM nodes use structured output into the Pydantic models of ``agentic_rag.agent.prompts``
  (plan decision 7) and format their prompts from the raw state inside the node.
- Failures: transient errors (for example an unreachable Ollama server) are retried by the
  node's ``RetryPolicy``; a failing tool becomes a ``SubtaskResult`` with ``ok=False``, so the
  answer can still be written from the other results; a structured reply that cannot be read
  falls back to a safe default and logs a warning (``analyze_request`` and ``plan_subtasks``
  search for the question, ``verify_answer`` marks verification unavailable); unexpected
  errors, such as a missing index, propagate.

Citations: the ``[n]`` markers of a RAG context, and so of a ``SubtaskResult.output``, are
local to one RAG run. They number that result's own ``sources``, so every ``retrieve``
sub-task starts again at ``[1]``. ``synthesize_answer``, the fan-in point, therefore assigns
one global numbering (:func:`numbered_sources`): the ``sources`` of ``subtask_results`` in
sub-task order, each result's in rank order, de-duplicated by ``chunk_id`` (a repeated chunk
keeps its first number). It builds its prompt from these numbered sources, not from the
per-run markers, and ``finalize_response`` returns exactly this list as ``sources``. The
answer's ``[n]`` is then ``sources[n - 1]``, the number the UI shows. Both nodes derive the
list from ``subtask_results`` with the same function, so their numbers agree.
"""

import json
import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any, Final

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage
from langchain_core.tools import BaseTool, ToolException
from langgraph.types import Overwrite
from pydantic import BaseModel, ValidationError

from agentic_rag.agent.prompts import (
    Plan,
    RequestAnalysis,
    Verification,
    analyze_messages,
    format_material,
    plan_messages,
    synthesize_messages,
    verify_messages,
)
from agentic_rag.agent.routing import is_exact_tool_answer
from agentic_rag.agent.state import AgentState, Subtask, SubtaskInput, SubtaskResult
from agentic_rag.agent.tools import SearchKnowledgeBaseTool
from agentic_rag.rag.state import Source
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

logger = logging.getLogger(__name__)

MAX_SUBTASKS: Final = 5
"""Size limit of a plan, so the fan-out of a round stays bounded."""

PARTIAL_NOTE: Final = "Note: this answer may be incomplete"
"""Start of the note ``finalize_response`` adds when the retries are exhausted."""

NO_DRAFT: Final = "I could not write an answer from the material that was found."
"""Draft used when the model returns no text."""

_THINKING: Final = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# A citation marker with the whitespace before it, so a dropped marker leaves no stray space.
_MARKER: Final = re.compile(r"\s*\[(\d+)\]")
_MAX_ERROR_LENGTH: Final = 300


def numbered_sources(results: Sequence[SubtaskResult]) -> list[Source]:
    """The global numbering of the citation contract: ``[n]`` is ``sources[n - 1]``.

    Args:
        results: The sub-task results of the round, in plan order.

    Returns:
        Their sources in sub-task order and, within a result, in rank order, without repeated
        chunks (a chunk keeps its first position).
    """
    seen: set[str] = set()
    sources: list[Source] = []
    for result in results:
        for source in result.sources:
            if source.chunk_id not in seen:
                seen.add(source.chunk_id)
                sources.append(source)
    return sources


def _summarize_analysis(update: dict[str, Any]) -> str:
    """Trace summary of ``analyze_request``."""
    subtasks = update.get("subtasks") or []
    detail = f": {subtasks[0].tool_name or subtasks[0].input}" if subtasks else ""
    return f"Intent {update.get('intent')}{detail}"


@traced(summarize=_summarize_analysis, metadata=lambda update: {"intent": update.get("intent")})
def analyze_request(
    state: AgentState, *, chat_model: BaseChatModel, tools: Mapping[str, BaseTool]
) -> dict[str, Any]:
    """Classify the latest user message and extract the question.

    Kind: LLM (structured output, :class:`~agentic_rag.agent.prompts.RequestAnalysis`). The
    first node of every run.

    Reads ``messages``: the latest user message, with up to three earlier exchanges as context.

    Writes ``question`` (a standalone question: references to earlier turns are resolved; the
    latest message itself when the model gives none) and ``intent`` on every route;
    ``AgentOutput`` requires both, and ``route_after_analyze`` routes by ``intent``. It writes
    ``language`` (the language of the message, named in English) when the model gives it, for
    ``synthesize_answer``. Depending on the intent it also writes:

    - ``direct``: ``draft_answer``, the reply to a request that needs neither retrieval nor a
      tool (a greeting, thanks, a question about the assistant; a question outside the topic
      goes to ``single``, where the search finds nothing and the answer says so);
      ``finalize_response`` delivers it. A ``direct`` intent without a reply becomes
      ``single``.
    - ``single``: ``subtasks`` with exactly one ``retrieve`` sub-task, whose input is the
      model's search query, or the question.
    - ``tool``: ``subtasks`` with exactly one ``tool`` sub-task, whose ``tool_name`` and
      ``tool_args`` come from the tool catalog. A tool name outside the catalog becomes
      ``complex``, so the planner decides.
    - ``complex``: nothing more; ``plan_subtasks`` decomposes the request.

    A reply that cannot be read as a ``RequestAnalysis`` falls back to ``single`` with the
    latest message as the query.

    Args:
        state: The graph state; only ``messages`` is guaranteed to be present.
        chat_model: The chat model, used with structured output.
        tools: The non-retrieval tools by name; their names, descriptions and argument schemas
            let the model recognize the ``tool`` intent and fill in the sub-task.

    Returns:
        The partial state update described above.
    """
    history, latest = _split_conversation(state["messages"])
    analysis = _structured(chat_model, RequestAnalysis, analyze_messages(history, latest, tools))
    if analysis is None:
        analysis = RequestAnalysis(intent="single")
    question = analysis.question.strip() or latest
    intent = analysis.intent
    if intent == "direct" and not analysis.reply.strip():
        intent = "single"
    if intent == "tool" and analysis.tool_name not in tools:
        logger.warning(
            "The analysis named an unknown tool %r; planning instead", analysis.tool_name
        )
        intent = "complex"

    update: dict[str, Any] = {"question": question, "intent": intent}
    if analysis.language.strip():
        update["language"] = analysis.language.strip()
    if intent == "direct":
        update["draft_answer"] = analysis.reply.strip()
    elif intent == "single":
        query = analysis.search_query.strip() or question
        update["subtasks"] = [Subtask(id="s1", kind="retrieve", input=query)]
    elif intent == "tool":
        update["subtasks"] = [
            Subtask(
                id="s1",
                kind="tool",
                input=question,
                tool_name=analysis.tool_name,
                tool_args=analysis.tool_args,
            )
        ]
    return update


def _summarize_plan(update: dict[str, Any]) -> str:
    """Trace summary of ``plan_subtasks``."""
    subtasks = update.get("subtasks", [])
    parts = [f"{subtask.tool_name or 'search'}: {subtask.input}" for subtask in subtasks]
    noun = "sub-task" if len(subtasks) == 1 else "sub-tasks"
    return f"{len(subtasks)} {noun}: " + "; ".join(parts)


@traced(summarize=_summarize_plan, metadata=lambda update: {"subtasks": len(update["subtasks"])})
def plan_subtasks(
    state: AgentState, *, chat_model: BaseChatModel, tools: Mapping[str, BaseTool]
) -> dict[str, Any]:
    """Decompose the question into independent sub-tasks, or re-plan after verification.

    Kind: LLM (structured output, :class:`~agentic_rag.agent.prompts.Plan`, plan decision 7).
    It runs for the ``complex`` intent, and again whenever ``route_after_verify`` sends a
    rejected draft back for a re-plan.

    Reads ``question``. On a re-plan (the last ``verdict`` is ``"insufficient"``) it also
    reads ``critique``, the previous ``subtasks`` and their ``subtask_results``: the
    successful results are kept, the model is told what they cover and what is missing, and
    it plans only the rest.

    Writes:

    - ``subtasks``: the new plan, which replaces the previous one. Every sub-task can run on its
      own: a ``retrieve`` sub-task carries a self-contained query, a ``tool`` sub-task the name
      of one of ``tools`` and its arguments (a sub-task that names an unknown tool becomes a
      search). Ids are unique within the run, also across the kept results. Repeated searches
      are dropped. The plan is never empty, because an empty fan-out would end the run
      without an answer (an empty or unreadable reply falls back to one ``retrieve`` sub-task
      for the question), and it has at most :data:`MAX_SUBTASKS` sub-tasks.
    - ``subtask_results``: ``langgraph.types.Overwrite(kept)``, which resets the append reducer
      at the start of every planning round. ``kept`` is ``[]`` in the first round and the
      successful results of the rejected round on a re-plan. The workers of the new round
      append to the reset list.

    Args:
        state: The graph state, with ``question`` set by ``analyze_request``.
        chat_model: The chat model, used with structured output.
        tools: The non-retrieval tools by name: the catalog a ``tool`` sub-task can name.

    Returns:
        The partial state update described above.
    """
    question = state.get("question") or _split_conversation(state["messages"])[1]
    replan = state.get("verdict") == "insufficient"
    kept = [result for result in state.get("subtask_results", []) if result.ok] if replan else []
    previous = {subtask.id: subtask for subtask in state.get("subtasks", [])}
    kept_subtasks = [
        previous[result.subtask_id] for result in kept if result.subtask_id in previous
    ]
    prompt = plan_messages(
        question,
        tools,
        max_subtasks=MAX_SUBTASKS,
        kept=kept_subtasks,
        critique=state.get("critique", "") if replan else "",
    )
    plan = _structured(chat_model, Plan, prompt) or Plan()

    used_ids = {result.subtask_id for result in kept}
    seen = {_signature(subtask) for subtask in kept_subtasks}
    subtasks: list[Subtask] = []
    for planned in plan.subtasks:
        if len(subtasks) == MAX_SUBTASKS:
            break
        if planned.kind == "tool" and planned.tool_name in tools:
            subtask = Subtask(
                id=_next_id(used_ids),
                kind="tool",
                input=planned.input.strip() or question,
                tool_name=planned.tool_name,
                tool_args=planned.tool_args,
            )
        else:
            query = planned.input.strip() or question
            subtask = Subtask(id=_next_id(used_ids), kind="retrieve", input=query)
        if _signature(subtask) in seen:
            continue  # the same search or tool call again: its result is kept or planned
        seen.add(_signature(subtask))
        used_ids.add(subtask.id)
        subtasks.append(subtask)
    if not subtasks:
        subtasks = [Subtask(id=_next_id(used_ids), kind="retrieve", input=question)]
    return {"subtasks": subtasks, "subtask_results": Overwrite(kept)}


def _summarize_result(update: dict[str, Any]) -> str:
    """Trace summary of a ``Send`` worker."""
    (result,) = update["subtask_results"]
    if not result.ok:
        return f"Failed: {result.error}"
    if result.kind == "retrieve":
        count = len(result.sources)
        return f"{count} source{'s' if count != 1 else ''} found"
    return result.output.strip().splitlines()[0] if result.output.strip() else "No output"


@traced(summarize=_summarize_result)
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
      docstring). A ``ToolException`` gives ``ok=False`` with a short ``error`` instead of
      failing the whole run; other errors (a missing index, an embedding mismatch, an
      unreachable model) propagate, because they are not specific to this sub-task.
    - ``trace``: the subgraph's own events (``RagOutput["trace"]``), forwarded so that the
      run's trace also holds the per-node latency of the RAG subgraph; ``traced`` appends this
      node's event after them.

    Args:
        state: The ``Send`` payload of one ``retrieve`` sub-task.
        search_tool: The ``search_knowledge_base`` tool, bound to the compiled RAG subgraph.

    Returns:
        The partial state update described above.
    """
    subtask = state["subtask"]
    call = {
        "type": "tool_call",
        "id": f"call_{subtask.id}",
        "name": search_tool.name,
        "args": {"query": subtask.input},
    }
    try:
        message = search_tool.invoke(call)
    except ToolException as exc:
        return {"subtask_results": [_failed(subtask, str(exc))]}
    output = message.artifact
    result = SubtaskResult(
        subtask_id=subtask.id,
        kind="retrieve",
        output=output["context"],
        sources=list(output["sources"]),
    )
    return {"subtask_results": [result], "trace": list(output.get("trace", []))}


@traced(summarize=_summarize_result)
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
    """
    subtask = state["subtask"]
    tool = tools.get(subtask.tool_name or "")
    if tool is None:
        return {"subtask_results": [_failed(subtask, f"unknown tool {subtask.tool_name!r}")]}
    try:
        output = tool.invoke(subtask.tool_args)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in issue['loc'])}: {issue['msg']}"
            for issue in exc.errors(include_url=False)
        )
        return {
            "subtask_results": [_failed(subtask, f"invalid arguments for {tool.name}: {problems}")]
        }
    except ToolException as exc:
        return {"subtask_results": [_failed(subtask, str(exc))]}
    result = SubtaskResult(
        subtask_id=subtask.id, kind="tool", output=str(output), tool_name=subtask.tool_name
    )
    return {"subtask_results": [result]}


@traced(summarize=lambda update: f"Draft of {len(update['draft_answer'])} characters")
def synthesize_answer(state: AgentState, *, chat_model: BaseChatModel) -> dict[str, Any]:
    """Write a draft answer from the results of the current planning round.

    Kind: LLM, except exact single-tool replies which make no model call. It runs once per
    round, after every ``Send`` worker of the round has finished
    (fan-in through the ``subtask_results`` reducer).

    Reads ``question`` and ``subtask_results`` (sources, tool outputs and failures of the
    current round). The prompt presents the retrieved chunks under the global numbers of the
    citation contract (see Citations in the module docstring) instead of the per-run markers of
    ``SubtaskResult.output``; a tool result contributes its ``output``, a failed step its
    error.

    Writes ``draft_answer``: an answer grounded only in the sub-task results that cites the
    retrieved chunks by those global ``[n]`` numbers; what the results do not cover is stated
    as unknown rather than guessed. The thinking block of a reasoning model is removed; an
    empty reply becomes :data:`NO_DRAFT`.

    Args:
        state: The graph state after the fan-in.
        chat_model: The chat model.

    Returns:
        The partial state update described above.
    """
    if is_exact_tool_answer(state):
        return {"draft_answer": ""}
    results = state.get("subtask_results", [])
    material = format_material(numbered_sources(results), results)
    latest = _split_conversation(state["messages"])[1]
    prompt = synthesize_messages(
        _question(state), material, message=latest, language=state.get("language", "")
    )
    reply = chat_model.invoke(prompt)
    draft = _THINKING.sub("", str(reply.text)).strip()
    return {"draft_answer": draft or NO_DRAFT}


@traced(
    summarize=lambda update: (
        f"Verdict {update['verdict']}"
        + (f": {update['critique']}" if update.get("critique") else "")
    ),
    metadata=lambda update: {"verdict": update["verdict"], "retry_count": update["retry_count"]},
)
def verify_answer(state: AgentState, *, chat_model: BaseChatModel) -> dict[str, Any]:
    """Check that the draft answer is supported by the sub-task results.

    Kind: LLM (structured output, :class:`~agentic_rag.agent.prompts.Verification`). Its
    verdict drives the bounded re-plan loop (``route_after_verify``).

    Reads ``question``, ``draft_answer`` and ``subtask_results``, plus ``verdict`` and
    ``retry_count`` of the previous verification, if there was one.

    Writes:

    - ``verdict``: ``"grounded"`` when the draft answers the question and every claim in it is
      supported by the results; ``"insufficient"`` otherwise. A reply that cannot be read
      returns ``"unavailable"`` and the final response withholds the unverified draft.
    - ``critique``: what is missing or unsupported, for the re-plan; empty when grounded.
    - ``retry_count``: the number of re-plans performed so far. It is 0 when the first draft is
      verified and grows by one for each draft that comes from a re-plan (that is, when the
      previous verdict was ``"insufficient"``). ``route_after_verify`` compares it with
      ``Settings.max_retries``.

    Args:
        state: The graph state, with ``draft_answer`` set by ``synthesize_answer``.
        chat_model: The chat model, used with structured output.

    Returns:
        The partial state update described above.
    """
    results = state.get("subtask_results", [])
    material = format_material(numbered_sources(results), results)
    prompt = verify_messages(_question(state), material, state.get("draft_answer", ""))
    verification = _structured(chat_model, Verification, prompt)
    previous = state.get("verdict")
    retry_count = state.get("retry_count", 0) + 1 if previous == "insufficient" else 0
    if verification is None:
        return {
            "verdict": "unavailable",
            "critique": "Verification returned an unreadable response. Please try again.",
            "retry_count": retry_count,
        }
    return {
        "verdict": "grounded" if verification.grounded else "insufficient",
        "critique": "" if verification.grounded else verification.missing.strip(),
        "retry_count": retry_count,
    }


@traced(summarize=lambda update: f"Answer with {len(update['sources'])} sources")
def finalize_response(state: AgentState) -> dict[str, Any]:
    """Turn the draft into the final answer, its sources and the assistant message.

    Kind: data (no LLM call). The last node of every run, reached on every route.

    Reads ``draft_answer`` (written by ``analyze_request`` on the ``direct`` route and by
    ``synthesize_answer`` otherwise), ``intent``, ``verdict``, ``critique`` and
    ``subtask_results``.

    Writes, on every route (``AgentOutput`` requires ``answer`` and ``sources``):

    - ``answer``: the draft, without citation markers that point past the end of ``sources``
      (and without the space before such a marker). The output of every successful tool call
      follows verbatim, under a ``Tool result`` heading with the tool's name, once per tool
      and output (a re-plan may run the same check again): the tools are deterministic, so
      their exact figures and verdicts stay visible even where a small model restates them
      wrongly.
      When the last verdict is still ``"insufficient"`` (the retries are exhausted), the
      answer ends with an explicit note (:data:`PARTIAL_NOTE`) that names what is missing.
      An ``unavailable`` verifier withholds the draft and asks the user to retry.
    - ``sources``: exactly the globally numbered list that ``synthesize_answer`` cited from
      (see Citations in the module docstring), so ``[n]`` in the answer is ``sources[n - 1]``;
      ``[]`` on the ``direct`` route. All retrieved chunks stay in the list, cited or not, so
      the UI shows the whole retrieved context.
    - ``messages``: the assistant message with the answer, appended by ``add_messages``.

    ``traced`` appends the final event of the run's trace.

    Args:
        state: The graph state.

    Returns:
        The partial state update described above.
    """
    direct = state.get("intent") == "direct"
    results = [] if direct else state.get("subtask_results", [])
    sources = numbered_sources(results)
    answer = _MARKER.sub(
        lambda match: match[0] if 1 <= int(match[1]) <= len(sources) else "",
        state.get("draft_answer", "") or NO_DRAFT,
    ).strip()
    if is_exact_tool_answer(state):
        answer = ""
    elif state.get("verdict") == "unavailable":
        answer = "I couldn't verify the answer, so I have withheld the draft. Please try again."
    names = {subtask.id: subtask.tool_name for subtask in state.get("subtasks", [])}
    shown: set[tuple[str, str]] = set()
    for result in results:
        if result.kind == "tool" and result.ok and result.output.strip():
            name = result.tool_name or names.get(result.subtask_id) or "tool"
            if (name, result.output.strip()) in shown:
                continue  # A re-plan ran the same check again; show its output once.
            shown.add((name, result.output.strip()))
            answer += f"\n\n**Tool result** (`{name}`):\n\n```text\n{result.output.strip()}\n```"
    if not direct and state.get("verdict") == "insufficient":
        critique = state.get("critique", "")
        note = f"{PARTIAL_NOTE}: {critique}" if critique else f"{PARTIAL_NOTE}."
        answer = f"{answer}\n\n*{note}*"
    answer = answer.strip()
    return {"answer": answer, "sources": sources, "messages": [AIMessage(content=answer)]}


def _split_conversation(messages: Sequence[AnyMessage]) -> tuple[list[AnyMessage], str]:
    """The earlier messages and the text of the latest user message."""
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].type == "human":
            return list(messages[:index]), str(messages[index].text).strip()
    return list(messages[:-1]), str(messages[-1].text).strip() if messages else ""


def _question(state: AgentState) -> str:
    """The standalone question, or the latest user message."""
    return state.get("question") or _split_conversation(state["messages"])[1]


def _structured[ModelT: BaseModel](
    chat_model: BaseChatModel, schema: type[ModelT], messages: list[BaseMessage]
) -> ModelT | None:
    """Ask for structured output; None, with a warning, when the reply cannot be read."""
    try:
        reply = chat_model.with_structured_output(schema).invoke(messages)
        return schema.model_validate(reply)
    except (OutputParserException, ValidationError) as exc:
        logger.warning(
            "The %s reply could not be read; using the fallback: %s", schema.__name__, exc
        )
        return None


def _signature(subtask: Subtask) -> str:
    """What a sub-task does, without its id: two sub-tasks with one signature repeat a step."""
    if subtask.kind == "retrieve":
        return f"retrieve:{subtask.input.casefold()}"
    return f"tool:{subtask.tool_name}:{json.dumps(subtask.tool_args, sort_keys=True, default=str)}"


def _next_id(used: set[str]) -> str:
    """The first ``s<n>`` id that is not used yet."""
    number = 1
    while f"s{number}" in used:
        number += 1
    return f"s{number}"


def _failed(subtask: Subtask, error: str) -> SubtaskResult:
    """A failed result with a short, single-line error."""
    message = " ".join(error.split())[:_MAX_ERROR_LENGTH] or "the step failed"
    return SubtaskResult(
        subtask_id=subtask.id,
        kind=subtask.kind,
        output="",
        ok=False,
        error=message,
        tool_name=subtask.tool_name,
    )
