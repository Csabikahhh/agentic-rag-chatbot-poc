"""Main agentic workflow: request analysis, planning, sub-task execution and verification.

Seven nodes (plan section 5.1) share ``AgentState``: ``analyze_request``, ``plan_subtasks``,
``run_rag_subtask``, ``call_tool``, ``synthesize_answer``, ``verify_answer`` and
``finalize_response``. Conditional edges route by intent and by verdict, and ``Send`` fans the
planned sub-tasks out to ``run_rag_subtask`` (which calls the RAG subgraph) and ``call_tool``.

Modules:

- ``types``: the literal types ``Intent``, ``Verdict`` and ``SubtaskKind``, free of LangGraph
  imports, for modules such as the evaluation that need only these names.
- ``state``: the state schemas and the records the nodes exchange; it re-exports ``types``.
- ``nodes``: the node functions.
- ``routing``: the conditional-edge functions and the ``Send`` fan-out.
- ``tools``: the retrieval tool and the non-retrieval tool(s).
- ``graph``: the ``StateGraph`` wiring and the compiled graph.
"""
