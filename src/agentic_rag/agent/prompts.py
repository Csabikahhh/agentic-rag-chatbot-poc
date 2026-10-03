"""Prompts and structured-output schemas of the main workflow's LLM nodes.

Four nodes call the chat model (plan section 5.1, decision 7):

- ``analyze_request`` -> :class:`RequestAnalysis` (structured): the intent, a standalone
  question, and per intent a reply, a search query or a tool call.
- ``plan_subtasks`` -> :class:`Plan` (structured): independent sub-tasks, each a search
  query or a tool call.
- ``synthesize_answer`` -> plain text: the answer, citing the excerpts by number.
- ``verify_answer`` -> :class:`Verification` (structured): grounded or not, and what is
  missing.

Each prompt is a system message with the instructions and one human message with the data.
The system messages start with fixed sentences (:data:`ANALYZE_INSTRUCTIONS` starts with
"You are the router", and so on), which the scripted fake model of
``agentic_rag.llm.DEFAULT_FAKE_RULES`` recognises; the human messages use fixed labels
(``Latest message:``, ``Question:``, ``Documentation excerpts:``, ``Tool results:``) for the
same reason. ``tests/test_agent_graph.py`` checks that the rules and the prompts still match.

The schemas keep every field optional except the decision itself, so that a small local model
that leaves a field out still produces a usable reply; the nodes fill in the gaps (for
example the question from the latest message). The JSON schema of each model is what Ollama's
structured output enforces.
"""

import json
from collections.abc import Mapping, Sequence
from typing import Any, Final

from langchain_core.messages import AnyMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from agentic_rag.agent.state import Intent, Subtask, SubtaskKind, SubtaskResult
from agentic_rag.rag.state import Source

__all__ = [
    "ANALYZE_INSTRUCTIONS",
    "MAX_HISTORY_MESSAGES",
    "PLAN_INSTRUCTIONS",
    "SYNTHESIZE_INSTRUCTIONS",
    "VERIFY_INSTRUCTIONS",
    "Plan",
    "PlannedSubtask",
    "RequestAnalysis",
    "Verification",
    "analyze_messages",
    "format_material",
    "plan_messages",
    "synthesize_messages",
    "tool_catalog",
    "verify_messages",
]

MAX_HISTORY_MESSAGES: Final = 6
"""Earlier messages shown to ``analyze_request``: the last three exchanges."""

ANALYZE_INSTRUCTIONS: Final = (
    "You are the router of a frontend developer assistant. The assistant answers from the "
    "official documentation of MDN Web Docs (HTML, CSS, JavaScript, accessibility), React, "
    "Vue, Next.js, Nuxt and TypeScript, and it can run these tools:\n"
    "{catalog}\n\n"
    "Classify the latest user message into one intent:\n"
    '- "direct": a greeting, thanks, or a question about the assistant itself. Write the '
    'complete reply in "reply", in the user\'s language.\n'
    '- "tool": a request that exactly one tool answers on its own: the contrast of two given '
    "colours (check_contrast), the specificity of given selectors (css_specificity), or "
    "whether a CSS, HTML, JavaScript or Web API feature works in given browsers "
    '(browser_support). Give "tool_name" and "tool_args" following the arguments of that '
    "tool. Never answer such a question from the documentation: the tool computes it.\n"
    '- "single": one question that the documentation answers with a single search. Write one '
    'English search query with the key terms in "search_query". A question outside web '
    'frontend development is "single" too: the search finds nothing, and the answer says so.\n'
    '- "complex": everything else: several questions, a comparison of frameworks or APIs, a '
    "question that needs the same tool twice, or one that needs both a tool and the "
    "documentation.\n"
    'Always write "question": the latest message as a standalone question in the user\'s '
    'language, with references to earlier messages resolved, and "language": the language '
    "of the latest message, named in English (for example Hungarian or English).\n\n"
    "Examples:\n"
    'Szia! Miben tudsz segíteni? -> {{"intent": "direct", "question": "Miben tudsz '
    'segíteni?", "language": "Hungarian", "reply": "Szia! Webes frontend-fejlesztési '
    "kérdésekben segítek az MDN, a React, a Vue, a Next.js, a Nuxt és a TypeScript "
    'dokumentációja alapján."}}\n'
    'Who won the 2018 World Cup? -> {{"intent": "single", "question": "Who won the 2018 World '
    'Cup?", "language": "English", "search_query": "2018 football World Cup winner"}}\n'
    'How does the useEffect cleanup work? -> {{"intent": "single", "question": "How does the '
    'useEffect cleanup work?", "language": "English", "search_query": "React useEffect '
    'cleanup function"}}\n'
    'Does #777 text on #fff pass AA? -> {{"intent": "tool", "question": "Does #777 text on '
    '#fff pass AA?", "language": "English", "tool_name": "check_contrast", "tool_args": '
    '{{"foreground": "#777", "background": "#fff"}}}}\n'
    'Is .btn.primary more specific than button:hover? -> {{"intent": "tool", "question": '
    '"Is .btn.primary more specific than button:hover?", "language": "English", '
    '"tool_name": "css_specificity", "tool_args": {{"selectors": [".btn.primary", '
    '"button:hover"]}}}}\n'
    'Which reads better, #333 on #fff or #777 on #000? -> {{"intent": "complex", "question": '
    '"Which reads better, #333 on #fff or #777 on #000?", "language": "English"}}\n'
    'Működik a :has() Safari 15-ben? -> {{"intent": "tool", "question": "Működik a :has() '
    'Safari 15-ben?", "language": "Hungarian", "tool_name": "browser_support", "tool_args": '
    '{{"feature": ":has()", "browsers": ["Safari 15"]}}}}\n'
    'Működik a :has() Safari 15-ben, és mivel helyettesíthetem? -> {{"intent": "complex", '
    '"question": "Működik a :has() Safari 15-ben, és mivel helyettesíthetem?", "language": '
    '"Hungarian"}}\n'
    'Mi a különbség a Nuxt useState és a React useState között? -> {{"intent": "complex", '
    '"question": "Mi a különbség a Nuxt useState és a React useState között?", "language": '
    '"Hungarian"}}\n\n'
    "Answer with a JSON object only."
)
"""System prompt of ``analyze_request``; ``{catalog}`` is :func:`tool_catalog`."""

PLAN_INSTRUCTIONS: Final = (
    "You split a request to a frontend documentation assistant into independent sub-tasks "
    "that run in parallel. Each sub-task is one of:\n"
    '- "retrieve": "input" is one English search query about one subject (one framework, '
    "API or concept);\n"
    '- "tool": "tool_name" and "tool_args" call one of these tools:\n'
    "{catalog}\n\n"
    "Rules:\n"
    "- To compare frameworks or APIs, plan one retrieve sub-task per framework or API.\n"
    "- Use a tool for what it computes (contrast ratios, specificity, browser support) "
    "instead of searching for it.\n"
    "- browser_support covers web platform features only (CSS, HTML, JavaScript, Web APIs), "
    "never React, Vue, Next.js or Nuxt APIs.\n"
    "- Plan between 1 and {max_subtasks} sub-tasks.\n\n"
    "Examples:\n"
    'Mi a különbség a Nuxt useState és a React useState között? -> {{"subtasks": [{{"kind": '
    '"retrieve", "input": "Nuxt useState composable shared state"}}, {{"kind": "retrieve", '
    '"input": "React useState hook"}}]}}\n'
    'Működik a :has() Safari 15-ben, és mivel helyettesíthetem? -> {{"subtasks": [{{"kind": '
    '"tool", "tool_name": "browser_support", "tool_args": {{"feature": ":has()", "browsers": '
    '["Safari 15"]}}}}, {{"kind": "retrieve", "input": "CSS :has() selector alternatives '
    'fallback"}}]}}\n\n'
    'Answer with a JSON object of the form {{"subtasks": [...]}} only.'
)
"""System prompt of ``plan_subtasks``; ``{catalog}`` and ``{max_subtasks}`` are filled in."""

SYNTHESIZE_INSTRUCTIONS: Final = (
    "You are a frontend developer assistant. Answer the question using only the numbered "
    "documentation excerpts and the tool results below.\n"
    "- Cite every statement taken from an excerpt with its number in square brackets, like "
    "[1] or [2][3]. Tool results need no citation; give their figures exactly.\n"
    "- Never invent numbers, versions or tool results: use only those in the material.\n"
    "- Tool results are exact. Repeat their verdicts (passes or fails, supported or not "
    "supported) unchanged, only translated: never turn a fail into a pass.\n"
    "- If the material does not answer part of the question, say so plainly instead of "
    "guessing.\n"
    "- Answer in the language of the user's message, even when the question below is in "
    "another language; keep code, API names and identifiers as written.\n"
    "- Be concise. Use a short code example from the excerpts when it helps."
)
"""System prompt of ``synthesize_answer``."""

VERIFY_INSTRUCTIONS: Final = (
    "You check a draft answer of a documentation assistant against the material it was "
    "written from.\n"
    "The draft is grounded when it answers the question and every claim in it is supported by "
    "the numbered excerpts or the tool results; saying that the material does not cover a "
    "point is acceptable.\n"
    "It is not grounded when it states facts the material does not contain (every number, "
    "version and ratio must appear in the material), contradicts the material, or leaves a "
    "part of the question unanswered that another search could cover.\n"
    'Reply with a JSON object {"grounded": true or false, "missing": "what is missing or '
    'unsupported; empty when grounded"} only.'
)
"""System prompt of ``verify_answer``."""


class RequestAnalysis(BaseModel):
    """Structured reply of ``analyze_request``."""

    model_config = ConfigDict(extra="ignore")

    intent: Intent = Field(description="The route: direct, single, tool or complex.")
    question: str = Field(
        default="",
        description="The latest message as a standalone question in the user's language.",
    )
    language: str = Field(
        default="",
        description="The language of the latest message, named in English, e.g. Hungarian.",
    )
    reply: str = Field(default="", description="For direct: the complete reply.")
    search_query: str = Field(default="", description="For single: one English search query.")
    tool_name: str | None = Field(default=None, description="For tool: the tool to call.")
    tool_args: dict[str, Any] = Field(
        default_factory=dict, description="For tool: the tool's arguments."
    )


class PlannedSubtask(BaseModel):
    """One sub-task of a :class:`Plan`, before the planner node gives it an id."""

    model_config = ConfigDict(extra="ignore")

    kind: SubtaskKind = Field(description="retrieve or tool.")
    input: str = Field(
        default="", description="For retrieve: one English search query; for tool: the goal."
    )
    tool_name: str | None = Field(default=None, description="For tool: the tool to call.")
    tool_args: dict[str, Any] = Field(
        default_factory=dict, description="For tool: the tool's arguments."
    )


class Plan(BaseModel):
    """Structured reply of ``plan_subtasks``."""

    model_config = ConfigDict(extra="ignore")

    subtasks: list[PlannedSubtask] = Field(
        default_factory=list, description="The independent sub-tasks."
    )


class Verification(BaseModel):
    """Structured reply of ``verify_answer``."""

    model_config = ConfigDict(extra="ignore")

    grounded: bool = Field(description="Whether the draft is supported and complete.")
    missing: str = Field(
        default="", description="What is missing or unsupported; empty when grounded."
    )


def tool_catalog(tools: Mapping[str, BaseTool]) -> str:
    """Describe the non-retrieval tools for a prompt: name, purpose and arguments.

    Args:
        tools: The tools by name.

    Returns:
        One line per tool, ``- name: description Arguments: {json schema properties}``, or a
        note that no tool is available.
    """
    if not tools:
        return "- (no tools)"
    lines = []
    for name, tool in tools.items():
        schema = tool.tool_call_schema.model_json_schema()
        arguments = {
            argument: {key: value for key, value in spec.items() if key != "title"}
            for argument, spec in schema.get("properties", {}).items()
        }
        lines.append(f"- {name}: {tool.description} Arguments: {json.dumps(arguments)}")
    return "\n".join(lines)


def analyze_messages(
    history: Sequence[AnyMessage], latest: str, tools: Mapping[str, BaseTool]
) -> list[BaseMessage]:
    """The prompt of ``analyze_request``.

    Args:
        history: The earlier messages of the conversation; the last
            :data:`MAX_HISTORY_MESSAGES` are shown.
        latest: The text of the latest user message.
        tools: The non-retrieval tools by name.

    Returns:
        A system and a human message.
    """
    earlier = [
        f"{'User' if message.type == 'human' else 'Assistant'}: {message.text}"
        for message in history[-MAX_HISTORY_MESSAGES:]
    ]
    conversation = "\n".join(earlier) if earlier else "(none)"
    return [
        SystemMessage(ANALYZE_INSTRUCTIONS.format(catalog=tool_catalog(tools))),
        HumanMessage(f"Earlier messages:\n{conversation}\n\nLatest message: {latest}"),
    ]


def plan_messages(
    question: str,
    tools: Mapping[str, BaseTool],
    *,
    max_subtasks: int,
    kept: Sequence[Subtask] = (),
    critique: str = "",
) -> list[BaseMessage]:
    """The prompt of ``plan_subtasks``; on a re-plan it names what is kept and what is missing.

    Args:
        question: The standalone question.
        tools: The non-retrieval tools by name.
        max_subtasks: The size limit of a plan.
        kept: The sub-tasks of the rejected round whose results the new round keeps.
        critique: What the verification found missing.

    Returns:
        A system and a human message.
    """
    system = PLAN_INSTRUCTIONS.format(catalog=tool_catalog(tools), max_subtasks=max_subtasks)
    parts = [f"Question: {question}"]
    if critique or kept:
        parts.append(
            "An earlier answer was rejected"
            + (f" because: {critique}" if critique else ".")
            + " Plan only the sub-tasks that obtain what is missing."
        )
    if kept:
        done = "\n".join(
            f"- {subtask.kind}: {subtask.input or subtask.tool_name}" for subtask in kept
        )
        parts.append(f"Already available, do not plan again:\n{done}")
    return [SystemMessage(system), HumanMessage("\n\n".join(parts))]


def format_material(sources: Sequence[Source], results: Sequence[SubtaskResult]) -> str:
    """The material of synthesis and verification: numbered excerpts, tool results, failures.

    Args:
        sources: The globally numbered sources; excerpt ``[n]`` is ``sources[n - 1]``.
        results: The sub-task results of the round, for the tool outputs and the failures.

    Returns:
        The material as labelled text blocks.
    """
    blocks = []
    tool_outputs = [result for result in results if result.kind == "tool" and result.ok]
    if sources:
        excerpts = "\n\n".join(
            f"[{number}] {source.content.strip()}" for number, source in enumerate(sources, 1)
        )
        blocks.append(f"Documentation excerpts:\n{excerpts}")
    elif not tool_outputs:
        # Only without tool results: next to a tool result, "none found" misleads the model
        # into calling the result unsupported.
        blocks.append("Documentation excerpts: none found.")
    if tool_outputs:
        blocks.append(
            "Tool results:\n" + "\n\n".join(f"- {result.output.strip()}" for result in tool_outputs)
        )
    failures = [result for result in results if not result.ok]
    if failures:
        blocks.append(
            "Failed steps:\n"
            + "\n".join(
                f"- {result.kind} {result.subtask_id}: {result.error}" for result in failures
            )
        )
    return "\n\n".join(blocks)


def synthesize_messages(
    question: str, material: str, *, message: str = "", language: str = ""
) -> list[BaseMessage]:
    """The prompt of ``synthesize_answer``.

    Args:
        question: The standalone question, which ``analyze_request`` may have translated.
        material: The output of :func:`format_material`.
        message: The user's own latest message; left out when it equals ``question``.
        language: The language of the answer, named in English (``Hungarian``). Small models
            follow a named language more reliably than "the language of the message".
    """
    asked = f"User's message: {message}\n" if message and message != question else ""
    answer_in = f"\n\nWrite the answer in {language}." if language else ""
    return [
        SystemMessage(SYNTHESIZE_INSTRUCTIONS),
        HumanMessage(f"{asked}Question: {question}\n\n{material}{answer_in}"),
    ]


def verify_messages(question: str, material: str, draft: str) -> list[BaseMessage]:
    """The prompt of ``verify_answer``."""
    return [
        SystemMessage(VERIFY_INSTRUCTIONS),
        HumanMessage(f"Question: {question}\n\n{material}\n\nDraft answer:\n{draft}"),
    ]
