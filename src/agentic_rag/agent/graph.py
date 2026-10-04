"""Wiring of the main agentic workflow (plan section 5.1).

:data:`NODE_NAMES` lists the seven nodes of the main graph; :func:`build_agent_graph` compiles
it. The RAG subgraph is not one of them: it runs inside ``run_rag_subtask``, through the
``search_knowledge_base`` tool, so it does not count towards the assignment's five nodes, and
``agentic-rag export-graph`` draws it as a diagram of its own.

Wiring::

    StateGraph(AgentState, input_schema=AgentInput, output_schema=AgentOutput)

    START             --> analyze_request
    analyze_request   --route_after_analyze--> finalize_response      (direct)
                                           --> plan_subtasks          (complex)
                                           --> Send: run_rag_subtask  (single)
                                           --> Send: call_tool        (tool)
    plan_subtasks     --dispatch_subtasks--->  Send per sub-task: run_rag_subtask | call_tool
    run_rag_subtask   --> synthesize_answer
    call_tool         --> synthesize_answer
    synthesize_answer --route_after_synthesize--> finalize_response (exact single tool)
                                              --> verify_answer (otherwise)
    verify_answer     --route_after_verify---> plan_subtasks          (insufficient and
                                                                       retry_count < max_retries)
                                           --> finalize_response      (otherwise)
    finalize_response --> END

The ``Send`` workers of a round run in parallel and fan in through the ``subtask_results``
reducer, so ``synthesize_answer`` runs once per round. The verify loop re-plans at most
``settings.max_retries`` times (0 disables it), always through the named node
``plan_subtasks``, which starts each round by resetting ``subtask_results``; the bound comes
from the settings, not from LangGraph's recursion limit. The conditional edges get explicit
path maps that match the routing annotations, because LangGraph cannot see ``Send`` targets,
and the LLM nodes and ``run_rag_subtask`` get a ``RetryPolicy`` for transient errors, such as
an Ollama server that is still starting.

Node convention, in both graphs: a node is a sync ``def`` whose only positional parameter is
the state (or its ``Send`` payload) and which returns a ``dict`` partial update or ``None``.
Its dependencies are keyword-only parameters, named so that they do not
clash with the arguments LangGraph injects (``config``, ``runtime``, ``writer``, ``store``,
``previous``). The graph builder creates them once, binds them with ``functools.partial`` and
registers each node under its explicit name; the ``Send`` workers also get
``input_schema=SubtaskInput``, because a partial hides the type hint LangGraph would infer it
from. Expensive resources, the vector store with its embedding model, are passed as
lock-guarded, lazily initialised zero-argument providers (``agentic_rag.rag.graph``). Building
a graph therefore loads nothing and stays offline, so ``export-graph`` can build it only to
draw it.

Execution model: the nodes are sync, and LangGraph runs the parallel ``Send`` workers of a
step in threads; the UI, the evaluation and the load test all take this synchronous path
(``docs/architecture.md``, *Execution model*). An ``async def`` node would break the UI's sync
``stream``.

Retries: :data:`RETRY_POLICY` (three attempts with LangGraph's default backoff) applies to the
four LLM nodes and to ``run_rag_subtask``. LangGraph's default ``retry_on`` retries connection
and server errors, not programming errors such as ``ValueError``.
"""

import functools
from typing import Final

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy

from agentic_rag.agent import nodes, routing
from agentic_rag.agent.state import AgentInput, AgentOutput, AgentState, SubtaskInput
from agentic_rag.agent.tools import SearchKnowledgeBaseTool, get_tools
from agentic_rag.config import Settings
from agentic_rag.llm import get_chat_model

__all__ = ["NODE_NAMES", "RETRY_POLICY", "build_agent_graph"]

RETRY_POLICY: Final = RetryPolicy(max_attempts=3)
"""Retry policy of the LLM nodes and of ``run_rag_subtask``: transient errors only."""

NODE_NAMES: Final[tuple[str, ...]] = (
    "analyze_request",
    "plan_subtasks",
    "run_rag_subtask",
    "call_tool",
    "synthesize_answer",
    "verify_answer",
    "finalize_response",
)
"""The seven nodes of the main graph, in the order of the table in plan section 5.1.

Each name is also the name of its node function in ``agentic_rag.agent.nodes`` and the
``node`` of its trace events.
"""


def build_agent_graph(
    settings: Settings,
) -> CompiledStateGraph[AgentState, None, AgentInput, AgentOutput]:
    """Compile the main agentic workflow.

    Args:
        settings: The settings to build the graph with. They select the chat model provider,
            the retrieval configuration of the RAG subgraph and the re-plan bound
            (``settings.max_retries``). Entry points pass ``get_settings()``.

    Returns:
        The compiled graph. It starts from an ``AgentInput`` (``{"messages": [...]}``) and
        returns the ``AgentOutput`` keys. Its dependencies are bound at build time, so
        ``invoke`` and ``stream`` need nothing but the input. Building loads no model and
        opens no index.
    """
    chat_model = get_chat_model(settings)
    search_tool, *other_tools = get_tools(settings)
    if not isinstance(search_tool, SearchKnowledgeBaseTool):
        msg = "get_tools must return the search tool first"
        raise TypeError(msg)
    tools = {tool.name: tool for tool in other_tools}

    builder = StateGraph(AgentState, input_schema=AgentInput, output_schema=AgentOutput)
    llm_bindings = {"chat_model": chat_model}
    builder.add_node(
        "analyze_request",
        functools.partial(nodes.analyze_request, **llm_bindings, tools=tools),
        retry_policy=RETRY_POLICY,
    )
    builder.add_node(
        "plan_subtasks",
        functools.partial(nodes.plan_subtasks, **llm_bindings, tools=tools),
        retry_policy=RETRY_POLICY,
    )
    builder.add_node(
        "run_rag_subtask",
        functools.partial(nodes.run_rag_subtask, search_tool=search_tool),
        input_schema=SubtaskInput,
        retry_policy=RETRY_POLICY,
    )
    builder.add_node(
        "call_tool", functools.partial(nodes.call_tool, tools=tools), input_schema=SubtaskInput
    )
    builder.add_node(
        "synthesize_answer",
        functools.partial(nodes.synthesize_answer, **llm_bindings),
        retry_policy=RETRY_POLICY,
    )
    builder.add_node(
        "verify_answer",
        functools.partial(nodes.verify_answer, **llm_bindings),
        retry_policy=RETRY_POLICY,
    )
    builder.add_node("finalize_response", nodes.finalize_response)

    builder.add_edge(START, "analyze_request")
    builder.add_conditional_edges(
        "analyze_request",
        routing.route_after_analyze,
        ["finalize_response", "plan_subtasks", "run_rag_subtask", "call_tool"],
    )
    builder.add_conditional_edges(
        "plan_subtasks", routing.dispatch_subtasks, ["run_rag_subtask", "call_tool"]
    )
    builder.add_edge("run_rag_subtask", "synthesize_answer")
    builder.add_edge("call_tool", "synthesize_answer")
    builder.add_conditional_edges(
        "synthesize_answer", routing.route_after_synthesize, ["finalize_response", "verify_answer"]
    )
    builder.add_conditional_edges(
        "verify_answer",
        functools.partial(routing.route_after_verify, max_retries=settings.max_retries),
        ["plan_subtasks", "finalize_response"],
    )
    builder.add_edge("finalize_response", END)
    return builder.compile(name="agent")
