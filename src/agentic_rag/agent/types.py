"""Literal types of the main workflow, free of LangGraph imports.

``agentic_rag.agent.state`` re-exports them. Modules that need only these names, such as the
evaluation and the load test, import them from here so that they do not load LangGraph.
"""

from typing import Literal

__all__ = ["Intent", "SubtaskKind", "Verdict"]

Intent = Literal["direct", "single", "complex", "tool"]
"""How ``analyze_request`` classifies a request; it selects the route after the node.

- ``direct``: answer without retrieval or tools (greetings, questions about the assistant).
- ``single``: one knowledge-base lookup, sent straight to ``run_rag_subtask``.
- ``complex``: a multi-part request that ``plan_subtasks`` decomposes into sub-tasks.
- ``tool``: one non-retrieval tool call, sent straight to ``call_tool``.
"""

Verdict = Literal["grounded", "insufficient"]
"""Outcome of ``verify_answer``.

- ``grounded``: the draft answer is supported by the sub-task results.
- ``insufficient``: unsupported or incomplete; re-plan while retries remain.
"""

SubtaskKind = Literal["retrieve", "tool"]
"""Kind of a sub-task: ``retrieve`` runs ``run_rag_subtask``, ``tool`` runs ``call_tool``."""
