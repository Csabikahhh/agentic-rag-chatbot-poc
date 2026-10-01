"""Modular RAG subgraph: query rewriting, retrieval, grading and context building.

The subgraph is separate from the main agentic workflow and does not count towards its
nodes. The main graph calls it from ``run_rag_subtask`` with an explicit input/output
mapping (``RagInput`` in, ``RagOutput`` out), so the two state schemas stay independent and
the subgraph can be tested and load-tested on its own.

Modules:

- ``state``: the ``Source`` record and the ``RagInput`` / ``RagOutput`` / ``RagState`` schemas.
- ``nodes``: ``rewrite_query``, ``retrieve``, ``grade_documents`` and ``build_context``.
- ``graph``: the compiled subgraph.
"""
