"""Conditional edges of the main agentic workflow (plan section 5.1).

Three routing points carry the workflow's autonomous decisions:

- after ``analyze_request``, :func:`route_after_analyze` routes by intent;
- after ``plan_subtasks``, :func:`dispatch_subtasks` fans the plan out, one ``Send`` per
  sub-task (decomposition into independently executed sub-tasks);
- after ``verify_answer``, :func:`route_after_verify` closes the bounded re-plan loop.

The return annotations list every possible destination, so the topology is documented in the
types. A routing function returns either the name of the next node (a ``Literal`` of node
names) or a fan-out (:data:`SubtaskSends`): a list of ``Send`` objects, each of which runs one
:data:`SubtaskWorker` node with its own :class:`~agentic_rag.agent.state.SubtaskInput`
instead of the shared state. LangGraph runs all sends of a step in parallel, and their results
fan in through the ``subtask_results`` reducer before ``synthesize_answer`` runs.

Routing functions are pure: they read the state (and, for the loop, the bound
``max_retries``), never call a model and never write to the state. LangGraph infers
destinations only from a plain ``Literal`` return annotation and cannot see a ``Send``'s
target, so ``build_agent_graph`` passes explicit path maps that match these annotations.

Skeleton: every function raises ``agentic_rag.errors.PlannedFeatureError`` until Phase 4
(plan section 8).
"""

from typing import Annotated, Literal

from langgraph.types import Send

from agentic_rag.agent.state import AgentState
from agentic_rag.errors import planned

__all__ = [
    "SubtaskSends",
    "SubtaskWorker",
    "dispatch_subtasks",
    "route_after_analyze",
    "route_after_verify",
]

SubtaskWorker = Literal["run_rag_subtask", "call_tool"]
"""The nodes a ``Send`` can target: ``run_rag_subtask`` runs ``retrieve`` sub-tasks and
``call_tool`` runs ``tool`` sub-tasks."""

SubtaskSends = Annotated[list[Send], SubtaskWorker]
"""A fan-out: one ``Send(worker, SubtaskInput(...))`` per sub-task, each to a SubtaskWorker.

Type checkers see ``list[Send]``. The ``Annotated`` metadata records the nodes the sends may
target, for readers, for ``build_agent_graph`` and for the tests
(``typing.get_type_hints(..., include_extras=True)``).
"""


def route_after_analyze(
    state: AgentState,
) -> Literal["finalize_response", "plan_subtasks"] | SubtaskSends:
    """Choose the path after ``analyze_request`` from ``intent``.

    - ``direct``: ``"finalize_response"``; ``analyze_request`` has already written the reply
      to ``draft_answer``, so retrieval, synthesis and verification are skipped.
    - ``complex``: ``"plan_subtasks"``, which decomposes the request.
    - ``single``: one ``Send`` to ``run_rag_subtask``.
    - ``tool``: one ``Send`` to ``call_tool``.

    For ``single`` and ``tool``, the one-step plan that ``analyze_request`` wrote to
    ``subtasks`` is dispatched exactly as :func:`dispatch_subtasks` dispatches a plan.

    Args:
        state: The graph state after ``analyze_request``.

    Returns:
        The next node, or the sends of the one-step plan.

    Raises:
        PlannedFeatureError: Always, until Phase 4.
    """
    raise planned(f"{__name__}.route_after_analyze", 4)


def dispatch_subtasks(state: AgentState) -> SubtaskSends:
    """Fan the current plan out: one ``Send`` per sub-task in ``subtasks``.

    A ``retrieve`` sub-task goes to ``run_rag_subtask`` and a ``tool`` sub-task to
    ``call_tool``. Each send carries ``SubtaskInput(subtask=..., question=state["question"])``.
    LangGraph runs all sends of the round in parallel, in one step; each worker appends its
    result to ``subtask_results``, and ``synthesize_answer`` runs once, after the last worker
    has finished. ``plan_subtasks`` never writes an empty plan, because an empty fan-out would
    leave no next node and end the run without an answer.

    Args:
        state: The graph state with the current plan in ``subtasks``.

    Returns:
        One send per sub-task, in plan order.

    Raises:
        PlannedFeatureError: Always, until Phase 4.
    """
    raise planned(f"{__name__}.dispatch_subtasks", 4)


def route_after_verify(
    state: AgentState, *, max_retries: int
) -> Literal["plan_subtasks", "finalize_response"]:
    """Close the bounded verify loop: accept the draft, re-plan, or give up.

    - ``verdict == "grounded"``: ``"finalize_response"``.
    - ``verdict == "insufficient"`` and ``retry_count < max_retries``: ``"plan_subtasks"``,
      which re-plans from the verdict and the rejected round's results.
    - Otherwise (the retries are exhausted): ``"finalize_response"``, which marks the answer as
      only partially answered.

    The loop therefore re-plans at most ``max_retries`` times per run, and it always passes
    through the named node ``plan_subtasks``, which starts the new round.

    Args:
        state: The graph state after ``verify_answer``.
        max_retries: ``Settings.max_retries``, bound by ``build_agent_graph``; 0 disables
            re-planning.

    Returns:
        The next node.

    Raises:
        PlannedFeatureError: Always, until Phase 4.
    """
    raise planned(f"{__name__}.route_after_verify", 4)
