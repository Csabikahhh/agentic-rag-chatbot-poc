"""Tool layer of the main agentic workflow (plan section 5.1, decisions 7 and 9).

The agent has two kinds of capability, each exposed as a LangChain tool:

- ``search_knowledge_base`` (:class:`SearchKnowledgeBaseTool`): the only retrieval tool. It
  runs the RAG subgraph for one query (``RagInput`` in, ``RagOutput`` out), and
  ``run_rag_subtask`` executes it for every ``retrieve`` sub-task.
- Three non-retrieval tools (decision 9), deterministic and local, which ``call_tool``
  executes for every ``tool`` sub-task, selected by the name the planner put in
  ``Subtask.tool_name``:

  - ``check_contrast`` (:class:`CheckContrastTool`): the WCAG 2.x contrast ratio of two CSS
    colours and the AA and AAA verdicts (``agentic_rag.agent.contrast``);
  - ``css_specificity`` (:class:`CssSpecificityTool`): the specificity of CSS selectors and
    which one wins (``agentic_rag.agent.specificity``);
  - ``browser_support`` (:class:`BrowserSupportTool`): the browser support of a web platform
    feature from MDN's browser-compat-data, optionally checked against target browsers
    (``agentic_rag.agent.compat``).

The model does not call the tools natively: the planner emits typed sub-tasks and explicit
nodes execute them (decision 7), which is more reliable with small local models. The tools
are still regular LangChain tools with a name, a description and an argument schema, so the
planner can describe them in its prompt (``tool_catalog`` in ``agentic_rag.agent.prompts``),
``call_tool`` invokes every tool the same way, and a tool-calling model can be bound to them
later (``bind_tools``) without restructuring.

Every non-retrieval tool returns plain text and raises ``ToolException`` for input it cannot
handle; ``call_tool`` turns that into a failed ``SubtaskResult``. :func:`get_tools` returns the
complete tool set for ``build_agent_graph``; building it loads nothing (the browser data is
read on the first call).
"""

from typing import Final, Literal, override

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.runnables import Runnable
from langchain_core.tools import ArgsSchema, BaseTool, ToolException
from pydantic import BaseModel, ConfigDict, Field

from agentic_rag.agent.compat import BCD_DIRECTORY, BROWSERS, CompatData, parse_target
from agentic_rag.agent.contrast import evaluate_contrast
from agentic_rag.agent.specificity import compare_specificity
from agentic_rag.config import Settings
from agentic_rag.rag.state import RagInput, RagOutput

__all__ = [
    "BROWSER_SUPPORT",
    "CHECK_CONTRAST",
    "CSS_SPECIFICITY",
    "SEARCH_KNOWLEDGE_BASE",
    "BrowserSupportInput",
    "BrowserSupportTool",
    "CheckContrastInput",
    "CheckContrastTool",
    "CssSpecificityInput",
    "CssSpecificityTool",
    "SearchKnowledgeBaseInput",
    "SearchKnowledgeBaseTool",
    "get_non_retrieval_tools",
    "get_tools",
]

SEARCH_KNOWLEDGE_BASE: Final = "search_knowledge_base"
"""Name of the retrieval tool, as models and tool messages see it."""

CHECK_CONTRAST: Final = "check_contrast"
"""Name of the WCAG contrast tool."""

CSS_SPECIFICITY: Final = "css_specificity"
"""Name of the CSS specificity tool."""

BROWSER_SUPPORT: Final = "browser_support"
"""Name of the browser-support tool."""


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

    ``_run`` calls ``rag_graph.invoke``; there is no ``_arun`` override. ``run_rag_subtask`` is
    a sync node, because the UI streams with the sync ``graph.stream``, and under ``ainvoke``
    LangGraph runs sync nodes in executor threads, so the tool's sync path is the one that
    runs. An async path (``_arun`` with ``rag_graph.ainvoke``) would need worker nodes with
    both a sync and an async implementation, for example ``RunnableLambda(func, afunc=...)``,
    or a UI that streams with ``astream``.

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
        """
        config = {"callbacks": run_manager.get_child()} if run_manager else None
        output = self.rag_graph.invoke({"query": query}, config=config)
        return output["context"], output


class CheckContrastInput(BaseModel):
    """Arguments of ``check_contrast``."""

    model_config = ConfigDict(extra="forbid")

    foreground: str = Field(
        min_length=1,
        description="The text or icon colour: hex such as #777 or #777777, rgb(), rgba(), "
        "hsl(), or a basic CSS colour name.",
    )
    background: str = Field(
        min_length=1, description="The opaque colour behind it, in the same formats."
    )


class CheckContrastTool(BaseTool):
    """WCAG 2.x contrast ratio of two colours, with the AA and AAA verdicts."""

    name: str = CHECK_CONTRAST
    description: str = (
        "Compute the WCAG 2.x contrast ratio of a foreground colour on a background colour and "
        "whether it passes AA and AAA for normal text, large text and UI components."
    )
    args_schema: ArgsSchema | None = CheckContrastInput

    @override
    def _run(
        self,
        foreground: str,
        background: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """Evaluate the contrast and describe it.

        Raises:
            ToolException: If a colour cannot be read or the background is transparent.
        """
        try:
            result = evaluate_contrast(foreground, background)
        except ValueError as exc:
            raise ToolException(str(exc)) from exc
        lines = [
            f"Contrast ratio {result.ratio:.2f}:1 for {result.foreground.hex} on "
            f"{result.background.hex} (WCAG 2.2, rounded down)."
        ]
        for level, what, minimum, passes in result.verdicts:
            verdict = "passes" if passes else "fails"
            lines.append(f"- {level} {what} (at least {minimum:g}:1): {verdict}")
        lines.append("Large text is at least 24px, or at least 18.66px and bold.")
        return "\n".join(lines)


class CssSpecificityInput(BaseModel):
    """Arguments of ``css_specificity``."""

    model_config = ConfigDict(extra="forbid")

    selectors: list[str] = Field(
        min_length=1,
        max_length=10,
        description="The CSS selectors to compare, one per item, written exactly as in the "
        "stylesheet, for example ['#nav .item', '.menu a:hover'].",
    )


class CssSpecificityTool(BaseTool):
    """Specificity of CSS selectors by Selectors Level 4, and which one wins."""

    name: str = CSS_SPECIFICITY
    description: str = (
        "Compute the specificity (a, b, c) of CSS selectors by the Selectors Level 4 rules, "
        "including :is(), :not(), :has() and :where(), and tell which selector wins."
    )
    args_schema: ArgsSchema | None = CssSpecificityInput

    @override
    def _run(
        self, selectors: list[str], run_manager: CallbackManagerForToolRun | None = None
    ) -> str:
        """Compute and compare the specificities.

        Raises:
            ToolException: If a selector cannot be parsed or uses the nesting selector.
        """
        try:
            pairs = compare_specificity(selectors)
        except ValueError as exc:
            raise ToolException(str(exc)) from exc
        lines = [
            "Specificity (a, b, c): a counts ID selectors; b classes, attribute selectors and "
            "pseudo-classes; c type selectors and pseudo-elements."
        ]
        lines += [f"- `{selector}`: {value}" for selector, value in pairs]
        if len(pairs) > 1:
            highest = max(value for _, value in pairs)
            winners = [selector for selector, value in pairs if value == highest]
            if len(winners) == 1:
                lines.append(f"`{winners[0]}` has the highest specificity {highest} and wins.")
            else:
                tied = ", ".join(f"`{selector}`" for selector in winners)
                lines.append(
                    f"{tied} share the highest specificity {highest}: the one declared later "
                    "in the stylesheet wins."
                )
        lines.append(
            "Specificity only decides between declarations of the same origin, importance and "
            "cascade layer: !important, layers and inline styles take precedence."
        )
        return "\n".join(lines)


class BrowserSupportInput(BaseModel):
    """Arguments of ``browser_support``."""

    model_config = ConfigDict(extra="forbid")

    feature: str = Field(
        min_length=1,
        description="The web platform feature as MDN names it, e.g. ':has()', '@container', "
        "'<dialog>', 'grid-template-areas', 'Promise.withResolvers', 'fetch', or a "
        "browser-compat-data id such as css.selectors.has.",
    )
    area: Literal["css", "html", "javascript", "api"] | None = Field(
        default=None, description="Where the feature belongs, when the name is ambiguous."
    )
    browsers: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Target browsers to check, each a name and a version, e.g. "
        "['Safari 15', 'Chrome 110']; empty for a summary only.",
    )


class BrowserSupportTool(BaseTool):
    """Browser support of a web platform feature, from MDN's browser-compat-data.

    Attributes:
        data: The downloaded browser-compat-data, loaded on the first call.
    """

    name: str = BROWSER_SUPPORT
    description: str = (
        "Look up which browser versions support a web platform feature (a CSS property, "
        "selector or at-rule, an HTML element or attribute, a JavaScript built-in or a Web "
        "API) in MDN's browser-compat-data, and check it against target browsers."
    )
    args_schema: ArgsSchema | None = BrowserSupportInput
    data: CompatData = Field(exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @override
    def _run(
        self,
        feature: str,
        area: Literal["css", "html", "javascript", "api"] | None = None,
        browsers: list[str] | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """Find the feature and describe its support.

        Raises:
            ToolException: If the data is missing, the feature is unknown or a target browser
                cannot be read.
        """
        try:
            matches = self.data.find(feature, area)
            targets = [parse_target(target) for target in browsers or []]
        except (FileNotFoundError, ValueError) as exc:
            raise ToolException(str(exc)) from exc
        if not matches:
            msg = (
                f"No feature named {feature!r} in MDN's browser-compat-data. Use the name as "
                "MDN writes it, such as ':has()', '<dialog>' or 'grid-template-areas', or a "
                "browser-compat-data id such as css.selectors.has."
            )
            raise ToolException(msg)
        best = matches[0]
        lines = [f"Browser support of {best.id} (MDN browser-compat-data):"]
        lines += [f"- {line}" for line in self.data.summary(best)]
        status = best.compat.get("status", {})
        flags = [
            name.replace("_", " ") for name in ("experimental", "deprecated") if status.get(name)
        ]
        if status.get("standard_track") is False:
            flags.append("non-standard")
        if flags:
            lines.append(f"Status: {', '.join(flags)}.")
        for browser, version in targets:
            verdict = self.data.check(best, browser, version)
            outcome = {True: "supported", False: "not supported", None: "unknown"}[
                verdict.supported
            ]
            lines.append(f"{BROWSERS[browser]} {version}: {outcome} ({verdict.detail}).")
        if best.compat.get("mdn_url"):
            lines.append(f"MDN: {best.compat['mdn_url']}")
        if len(matches) > 1:
            others = ", ".join(match.id for match in matches[1:4])
            lines.append(f"Other features with this name: {others}.")
        return "\n".join(lines)


def get_non_retrieval_tools(settings: Settings) -> list[BaseTool]:
    """Return the agent's non-retrieval tools: contrast, specificity and browser support.

    Every tool is deterministic and local: no network access and no model call inside the
    tool. Each has a precise argument schema, so the planner can fill in ``Subtask.tool_args``
    and the arguments are validated before the tool runs; each returns text, which becomes
    ``SubtaskResult.output``, and raises ``ToolException`` for input it cannot handle.

    Args:
        settings: The settings; the browser-support tool reads
            ``settings.data_dir / "browser-compat-data"`` on its first call.

    Returns:
        ``[check_contrast, css_specificity, browser_support]``.
    """
    return [
        CheckContrastTool(),
        CssSpecificityTool(),
        BrowserSupportTool(data=CompatData(settings.data_dir / BCD_DIRECTORY)),
    ]


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
    """
    if rag_graph is None:
        from agentic_rag.rag.graph import build_rag_graph

        rag_graph = build_rag_graph(settings)
    return [SearchKnowledgeBaseTool(rag_graph=rag_graph), *get_non_retrieval_tools(settings)]
