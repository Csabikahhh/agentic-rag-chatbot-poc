"""Tool layer of the main agentic workflow (plan section 5.1, decisions 7 and 9).

The agent has two kinds of capability, each exposed as a LangChain tool:

- ``search_knowledge_base`` (:class:`SearchKnowledgeBaseTool`): the only retrieval tool. It
  runs the RAG subgraph for one query (``RagInput`` in, ``RagOutput`` out), and
  ``run_rag_subtask`` executes it for every ``retrieve`` sub-task.
- The non-retrieval tool: deterministic and local, chosen together with the domain and the
  corpus (decisions 8 and 9), which are still open. :func:`get_non_retrieval_tools` is its
  placeholder. ``call_tool`` executes it for every ``tool`` sub-task, selected by the name the
  planner put in ``Subtask.tool_name``.

The model does not call the tools natively: the planner emits typed sub-tasks and explicit
nodes execute them (decision 7), which is more reliable with small local models. The tools
are still regular LangChain tools with a name, a description and an argument schema, so the
planner can describe them in its prompt, ``call_tool`` invokes every tool the same way, and a
tool-calling model can be bound to them later (``bind_tools``) without restructuring.

:func:`get_tools` returns the complete tool set for ``build_agent_graph``.

Skeleton: the name, description and argument schema of the search tool are final; executing
it and building the tool set raise ``agentic_rag.errors.PlannedFeatureError`` until Phase 4
(plan section 8).
"""

from typing import Final, Literal, override

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.runnables import Runnable
from langchain_core.tools import ArgsSchema, BaseTool
from pydantic import BaseModel, Field

from agentic_rag.config import Settings
from agentic_rag.errors import planned
from agentic_rag.rag.state import RagInput, RagOutput

__all__ = [
    "SEARCH_KNOWLEDGE_BASE",
    "SearchKnowledgeBaseInput",
    "SearchKnowledgeBaseTool",
    "get_non_retrieval_tools",
    "get_tools",
]

SEARCH_KNOWLEDGE_BASE: Final = "search_knowledge_base"
"""Name of the retrieval tool, as models and tool messages see it."""


class SearchKnowledgeBaseInput(BaseModel):
    """Arguments of ``search_knowledge_base``: the schema a tool-calling model fills in."""

    query: str = Field(
        min_length=1,
        description="A self-contained search query that names its subject explicitly "
        "instead of referring to earlier messages.",
    )


class SearchKnowledgeBaseTool(BaseTool):
    """The retrieval tool: answers one query with the RAG subgraph.

    The tool uses ``response_format="content_and_artifact"``:

    - content: ``RagOutput["context"]``, the retrieved text with citation markers; this is what
      a tool-calling model sees.
    - artifact: the whole ``RagOutput`` (``context``, ``sources`` and ``trace``) for the
      application. Invoking the tool with a tool call (``{"type": "tool_call", "id": ...,
      "name": ..., "args": {...}}``) returns a ``ToolMessage`` that carries both;
      ``run_rag_subtask`` does this to cite the sources and to forward the subgraph's trace.

    Phase 4 implements ``_run`` with ``rag_graph.invoke`` and does not override ``_arun``.
    ``run_rag_subtask`` is a sync node, because the UI streams with the sync
    ``graph.stream``, and under ``ainvoke`` LangGraph runs sync nodes in executor threads, so
    the tool's sync path is the one that runs. An async path (``_arun`` with
    ``rag_graph.ainvoke``) would need worker nodes with both a sync and an async
    implementation, for example ``RunnableLambda(func, afunc=...)``, or a UI that streams
    with ``astream``.

    Attributes:
        rag_graph: The compiled RAG subgraph (``agentic_rag.rag.graph.build_rag_graph``), or any
            runnable with the same ``RagInput`` -> ``RagOutput`` contract, for example a fake in
            tests. It is excluded from serialization.
    """

    name: str = SEARCH_KNOWLEDGE_BASE
    description: str = (
        "Search the knowledge base for passages relevant to a query. Returns the passages "
        "with citation markers that refer to their sources."
    )
    args_schema: ArgsSchema | None = SearchKnowledgeBaseInput
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"
    rag_graph: Runnable[RagInput, RagOutput] = Field(
        exclude=True, description="The RAG subgraph that answers the queries."
    )

    @override
    def _run(
        self, query: str, run_manager: CallbackManagerForToolRun | None = None
    ) -> tuple[str, RagOutput]:
        """Run the RAG subgraph for one query.

        Args:
            query: The search query, already validated by :class:`SearchKnowledgeBaseInput`.
            run_manager: LangChain's callback manager for this tool run.

        Returns:
            ``(content, artifact)``: the context text and the whole ``RagOutput``.

        Raises:
            PlannedFeatureError: Always, until Phase 4.
        """
        raise planned(f"{__name__}.SearchKnowledgeBaseTool._run", 4)


def get_non_retrieval_tools(settings: Settings) -> list[BaseTool]:
    """Return the agent's non-retrieval tool(s); a placeholder until the domain is chosen.

    The assignment requires at least one tool that does something other than retrieval. The
    right tool depends on the domain and the corpus (plan decisions 8 and 9), which are still
    open, so no tool is implemented or named here yet. The chosen tool must:

    - be deterministic and local: no network access and no model call inside the tool;
    - be a LangChain tool (an ``@tool`` function or a ``BaseTool`` subclass) with a precise
      argument schema, so the planner can fill in ``Subtask.tool_args`` and the arguments are
      validated before the tool runs;
    - return text, which becomes ``SubtaskResult.output``, and raise ``ToolException`` for
      input it cannot handle, which ``call_tool`` turns into a failed result;
    - come with unit tests of its own.

    Args:
        settings: The settings, for a tool that needs configuration (for example a data file
            under ``settings.data_dir``).

    Returns:
        The non-retrieval tools, with unique names other than ``search_knowledge_base``.

    Raises:
        PlannedFeatureError: Always, until the tool is chosen and built in Phase 4.
    """
    raise planned(f"{__name__}.get_non_retrieval_tools", 4)


def get_tools(
    settings: Settings, *, rag_graph: Runnable[RagInput, RagOutput] | None = None
) -> list[BaseTool]:
    """Return every tool of the agent: ``search_knowledge_base`` first, then the others.

    ``build_agent_graph`` binds the search tool to ``run_rag_subtask``, and the non-retrieval
    tools, by name, to ``analyze_request``, ``plan_subtasks`` and ``call_tool``.

    Args:
        settings: The settings to build the tools with.
        rag_graph: The RAG subgraph for the search tool. ``None`` builds it with
            ``agentic_rag.rag.graph.build_rag_graph(settings)``, imported inside the function.
            Tests pass a fake.

    Returns:
        ``[SearchKnowledgeBaseTool(rag_graph=...), *get_non_retrieval_tools(settings)]``; the
        tool names are unique.

    Raises:
        PlannedFeatureError: Always, until Phase 4.
    """
    raise planned(f"{__name__}.get_tools", 4)
