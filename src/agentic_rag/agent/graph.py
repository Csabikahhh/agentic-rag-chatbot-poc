"""Wiring of the main agentic workflow (plan section 5.1).

:data:`NODE_NAMES` lists the seven nodes of the main graph; :func:`build_agent_graph` compiles
it. The RAG subgraph is not one of them: it runs inside ``run_rag_subtask``, through the
``search_knowledge_base`` tool, so it does not count towards the assignment's five nodes, and
``agentic-rag export-graph`` draws it as a diagram of its own.

Planned wiring (Phase 4)::

    StateGraph(AgentState, input_schema=AgentInput, output_schema=AgentOutput)

    START             --> analyze_request
    analyze_request   --route_after_analyze--> finalize_response      (direct)
                                           --> plan_subtasks          (complex)
                                           --> Send: run_rag_subtask  (single)
                                           --> Send: call_tool        (tool)
    plan_subtasks     --dispatch_subtasks--->  Send per sub-task: run_rag_subtask | call_tool
    run_rag_subtask   --> synthesize_answer
    call_tool         --> synthesize_answer
    synthesize_answer --> verify_answer
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
Its dependencies are keyword-only parameters without defaults, named so that they do not
clash with the arguments LangGraph injects (``config``, ``runtime``, ``writer``, ``store``,
``previous``). The graph builder creates them once, binds them with ``functools.partial`` and
registers each node under its explicit name; the ``Send`` workers also get
``input_schema=SubtaskInput``, because a partial hides the type hint LangGraph would infer it
from. Expensive resources, the vector store with its embedding model, are passed as
lock-guarded, lazily initialised zero-argument providers (a bare ``functools.cache`` is not
enough: concurrent first calls would each build the resource). Building a graph therefore
loads nothing and stays offline, so ``export-graph`` can build it only to draw it.

Execution model: the nodes are sync. The UI streams with ``graph.stream(...,
stream_mode=["updates", "values"], version="v2")``, the load test calls ``graph.invoke`` from a
``ThreadPoolExecutor(max_workers=concurrency)``, and LangGraph runs the parallel ``Send``
workers of a step in threads. An ``async def`` node would break the sync ``stream``, which
raises ``TypeError`` for a node without a sync implementation.

Skeleton: :data:`NODE_NAMES` is final; :func:`build_agent_graph` raises
``agentic_rag.errors.PlannedFeatureError`` until Phase 4 (plan section 8).
"""

from typing import Final

from langgraph.graph.state import CompiledStateGraph

from agentic_rag.agent.state import AgentInput, AgentOutput, AgentState
from agentic_rag.config import Settings
from agentic_rag.errors import planned

__all__ = ["NODE_NAMES", "build_agent_graph"]

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
        ``invoke`` and ``stream`` need nothing but the input.

    Raises:
        PlannedFeatureError: Always, until Phase 4.
    """
    raise planned(f"{__name__}.build_agent_graph", 4)
