"""Chat model factory: a local model served by Ollama, or a scripted offline fake.

``get_chat_model(settings)`` is the only place that creates chat models, so every node uses the
provider chosen by ``LLM_PROVIDER`` (plan decision 3):

- ``ollama``: ``langchain_ollama.ChatOllama`` with the model, URL, context window
  (``OLLAMA_NUM_CTX``), request timeout (``OLLAMA_TIMEOUT_S``) and temperature from the
  settings. Constructing it does not contact the server; the first call does.
- ``fake``: a ``ScriptedChatModel``, deterministic and offline. It is the assignment's "dummy
  LLM" fallback and keeps tests, CI and UI work free of any model.

The fake model answers from an ordered list of rules (``FakeRule``): the first rule whose
regular expression matches the prompt text wins, otherwise it returns a fixed default reply. Its
``with_structured_output()`` parses the reply as JSON into a Pydantic model, so nodes can use
structured output (decision 7) with both providers. A rule with ``expand=True`` fills the
groups of its match into its reply (``match.expand``), so a rule can copy parts of the prompt,
such as two colours of the question, into a JSON reply.

``DEFAULT_FAKE_RULES`` is the rule set of fake mode. It recognises the four prompts of the main
workflow by their opening sentences (``agentic_rag.agent.prompts``) and scripts a plausible,
deterministic run: greetings get a direct reply, a question with two hex colours, with two
selectors and the word *specificity*, or with a feature and a browser version goes to the
matching tool, a question with *and* / *vs* / *difference* is split into two searches, any
other question becomes one search; the synthesis cites source [1] (or repeats a tool's
output), and every draft is accepted. It understands no language: it only shows the
workflow's paths without a model.

The graph nodes are sync functions: the UI streams the graph with ``graph.stream`` and the
load test runs ``graph.invoke`` in a thread pool. The fake therefore implements the sync paths
only (``invoke``, ``batch`` and token streaming with ``stream``); its async methods use
LangChain's default fallbacks, which run the sync code in a thread pool.
"""

import logging
import re
from collections.abc import Iterator, Sequence
from operator import itemgetter
from typing import Any, override

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda, RunnableMap, RunnablePassthrough
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError, field_validator

from agentic_rag.config import Settings

logger = logging.getLogger(__name__)

FAKE_MODEL_NAME = "scripted-fake"
"""Model name the fake reports in ``response_metadata["model_name"]``."""

DEFAULT_FAKE_REPLY = (
    "This is a scripted reply from the fake LLM provider (LLM_PROVIDER=fake); "
    "no language model was called."
)
"""Reply of the fake model when no rule matches."""

# Word-sized pieces (a word plus its trailing whitespace) that add up to the whole reply.
_TOKEN_PATTERN = re.compile(r"\S+\s*|\s+")


class FakeRule(BaseModel):
    """One scripted answer of the fake chat model.

    Attributes:
        pattern: Regular expression searched (``re.search``) in the prompt text. Use inline
            flags for options: ``(?i)`` for case-insensitive matching, ``(?s)`` to let ``.``
            match across lines and messages.
        reply: The reply returned when ``pattern`` matches.
    """

    model_config = ConfigDict(frozen=True)

    pattern: str = Field(description="Regular expression searched in the prompt text.")
    reply: str = Field(description="Reply returned when the pattern matches.")
    expand: bool = Field(
        default=False,
        description="Fill the match's groups into the reply with re.Match.expand: \\1 or "
        "\\g<name> in the reply become the matched text.",
    )

    @field_validator("pattern")
    @classmethod
    def _check_pattern(cls, value: str) -> str:
        """Fail when the rule is defined, not when it is first used."""
        try:
            re.compile(value)
        except re.error as exc:
            msg = f"invalid regular expression {value!r}: {exc}"
            raise ValueError(msg) from exc
        return value

    def matches(self, prompt: str) -> bool:
        """Return whether the rule applies to a prompt.

        Args:
            prompt: The prompt text, as built by ``render_prompt()``.

        Returns:
            True when ``pattern`` is found anywhere in ``prompt``.
        """
        return re.search(self.pattern, prompt) is not None


_ANALYZE = r"(?s)\AYou are the router of a frontend developer assistant\..*?\n\nLatest message: "
_PLAN = r"(?s)\AYou split a request"
_SYNTHESIZE = r"(?s)\AYou are a frontend developer assistant\. Answer"
_VERIFY = r"(?s)\AYou check a draft answer"

DEFAULT_FAKE_RULES: tuple[FakeRule, ...] = (
    FakeRule(
        pattern=_ANALYZE + r"(?i:hi|hello|hey|szia|helló|thanks|thank you|köszönöm)\b",
        reply='{"intent": "direct", "reply": "Hi! I answer questions about web frontend '
        "development from the official MDN, React, Vue, Next.js, Nuxt and TypeScript "
        'documentation, and I can check colour contrast, CSS specificity and browser support."}',
    ),
    FakeRule(
        pattern=_ANALYZE
        + r"[^\n]*?(?P<fg>#[0-9a-fA-F]{3,8})\b"
        + r"[^\n]*?(?P<bg>#[0-9a-fA-F]{3,8}\b|(?i:white|black)\b)",
        reply='{"intent": "tool", "tool_name": "check_contrast", '
        '"tool_args": {"foreground": "\\g<fg>", "background": "\\g<bg>"}}',
        expand=True,
    ),
    FakeRule(
        pattern=_ANALYZE
        + r"(?=[^\n]*(?i:specificity|specificitás))[^`\n]*`(?P<a>[^`\n]+)`[^`\n]*`(?P<b>[^`\n]+)`",
        reply='{"intent": "tool", "tool_name": "css_specificity", '
        '"tool_args": {"selectors": ["\\g<a>", "\\g<b>"]}}',
        expand=True,
    ),
    FakeRule(
        pattern=_ANALYZE
        + r"(?=[^\n]*(?i:support|működik|támogat))[^`\n]*`(?P<feature>[^`\n]+)`[^\n]*?"
        r"(?P<browser>(?i:safari|chrome|firefox|edge))\s*(?P<version>\d+(?:\.\d+)?)",
        reply='{"intent": "tool", "tool_name": "browser_support", "tool_args": '
        '{"feature": "\\g<feature>", "browsers": ["\\g<browser> \\g<version>"]}}',
        expand=True,
    ),
    FakeRule(
        pattern=_ANALYZE + r"[^\n]*\b(?i:and|vs|versus|compare|difference|különbség\w*|és)\b",
        reply='{"intent": "complex"}',
    ),
    FakeRule(pattern=_ANALYZE, reply='{"intent": "single"}'),
    FakeRule(
        pattern=_PLAN
        + r".*?\n\nQuestion: (?P<a>[^\n]+?)\s+(?i:and|vs\.?|versus|és)\s+(?P<b>[^\n?]+)",
        reply='{"subtasks": [{"kind": "retrieve", "input": "\\g<a>"}, '
        '{"kind": "retrieve", "input": "\\g<b>"}]}',
        expand=True,
    ),
    FakeRule(pattern=_PLAN, reply='{"subtasks": []}'),
    FakeRule(
        pattern=_SYNTHESIZE + r".*?\nDocumentation excerpts:\n\[1\] ",
        reply="This scripted answer stands in for a language model (LLM_PROVIDER=fake): the "
        "most relevant passage of the documentation is source [1], shown with the other "
        "retrieved passages below.",
    ),
    FakeRule(
        pattern=_SYNTHESIZE
        + r".*?\nTool results:\n- (?P<output>.+?)(?=\n\n(?:Failed steps:|Write the answer)|\Z)",
        reply="\\g<output>\n\n(Scripted answer of the fake LLM provider; the figures come "
        "from the tool.)",
        expand=True,
    ),
    FakeRule(
        pattern=_SYNTHESIZE,
        reply="The documentation that was searched does not cover this question. (Scripted "
        "answer of the fake LLM provider.)",
    ),
    FakeRule(pattern=_VERIFY, reply='{"grounded": true, "missing": ""}'),
)
"""Rules of fake mode, used by ``get_chat_model()``; see the module docstring."""


def render_prompt(messages: Sequence[BaseMessage]) -> str:
    """Return the prompt text that the fake model matches its rules against and records.

    The text of every message, in order, joined by a blank line. A plain string input becomes
    a single human message, so its prompt text is the string itself.

    Args:
        messages: The messages the chat model received.

    Returns:
        The prompt text.
    """
    return "\n\n".join(str(message.text) for message in messages)


class ScriptedChatModel(BaseChatModel):
    """Deterministic, offline chat model that answers from scripted rules.

    The first rule in ``rules`` whose pattern matches the prompt text (see ``render_prompt()``)
    provides the reply; when none matches, the reply is ``default_reply``. Every call records
    its prompt text, readable through ``prompts`` and ``last_prompt``. ``invoke``, ``batch``
    and ``stream`` are implemented natively; streaming yields the reply in word-sized chunks.
    The async methods work through LangChain's thread-pool fallbacks, since no node is async.
    ``with_structured_output()`` parses the reply as JSON into a Pydantic model. Stop
    sequences are ignored.

    Example:
        >>> model = ScriptedChatModel(rules=[FakeRule(pattern="(?i)capital", reply="Paris")])
        >>> model.invoke("What is the capital of France?").content
        'Paris'
        >>> model.prompts
        ('What is the capital of France?',)

    Attributes:
        rules: Ordered rules; the first one whose pattern matches the prompt text wins.
        default_reply: The reply when no rule matches.
    """

    rules: list[FakeRule] = Field(
        default_factory=list,
        description="Ordered rules; the first one whose pattern matches the prompt text wins.",
    )
    default_reply: str = Field(
        default=DEFAULT_FAKE_REPLY,
        description="Reply when no rule matches.",
    )

    # Appending to and copying a list are atomic in CPython, so the threads of parallel graph
    # branches and of the load test can record prompts without a lock.
    _prompts: list[str] = PrivateAttr(default_factory=list)

    @property
    def prompts(self) -> tuple[str, ...]:
        """The prompt texts received so far, oldest first, as an immutable snapshot."""
        return tuple(self._prompts)

    @property
    def last_prompt(self) -> str | None:
        """The most recent prompt text, or None before the first call."""
        prompts = self.prompts
        return prompts[-1] if prompts else None

    def clear_prompts(self) -> None:
        """Forget the recorded prompts."""
        self._prompts.clear()

    def reply_for(self, prompt: str) -> str:
        """Return the reply the rules give for a prompt text, without recording it.

        Args:
            prompt: The prompt text, as built by ``render_prompt()``.

        Returns:
            The reply of the first matching rule, or ``default_reply``.
        """
        for rule in self.rules:
            match = re.search(rule.pattern, prompt)
            if match is not None:
                return match.expand(rule.reply) if rule.expand else rule.reply
        return self.default_reply

    def _respond(self, messages: Sequence[BaseMessage]) -> str:
        """Record the prompt of one call and return its reply."""
        prompt = render_prompt(messages)
        self._prompts.append(prompt)
        return self.reply_for(prompt)

    @property
    @override
    def _llm_type(self) -> str:
        return "scripted-fake-chat-model"

    @property
    @override
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": FAKE_MODEL_NAME, "rules": [rule.pattern for rule in self.rules]}

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return _chat_result(self._respond(messages))

    @override
    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        yield from _reply_chunks(self._respond(messages))

    @override
    def with_structured_output(
        self,
        schema: dict[str, Any] | type,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, dict[str, Any] | BaseModel]:
        """Return a runnable that parses the scripted reply into a Pydantic model.

        This mirrors ``ChatOllama.with_structured_output(schema)`` (JSON schema mode): the
        reply text must be a JSON object that validates against ``schema``, and a reply that
        does not raises ``OutputParserException`` when the runnable is invoked, as with Ollama.

        Args:
            schema: A Pydantic model class.
            include_raw: If True, the runnable returns a dict with the keys ``raw`` (the
                AIMessage), ``parsed`` (the model instance, or None) and ``parsing_error`` (the
                exception, or None) instead of raising on a bad reply.
            **kwargs: ``method`` and ``strict`` are accepted for API parity with ChatOllama and
                ignored.

        Returns:
            A runnable that takes the chat model's input and returns an instance of
            ``schema``, or the dict described under ``include_raw``.

        Raises:
            TypeError: If ``schema`` is not a Pydantic model class.
            ValueError: If an unsupported keyword argument is passed.
        """
        kwargs.pop("method", None)
        kwargs.pop("strict", None)
        if kwargs:
            msg = f"Received unsupported arguments {kwargs}"
            raise ValueError(msg)
        if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            msg = (
                "ScriptedChatModel.with_structured_output supports Pydantic model classes "
                f"only, got {schema!r}"
            )
            raise TypeError(msg)
        model_class: type[BaseModel] = schema

        def parse(message: BaseMessage) -> BaseModel:
            return _parse_structured_reply(message, model_class)

        parser = RunnableLambda(parse, name=f"Parse{model_class.__name__}")
        if not include_raw:
            return self | parser
        parser_assign = RunnablePassthrough.assign(
            parsed=itemgetter("raw") | parser, parsing_error=lambda _: None
        )
        parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
        parser_with_fallback = parser_assign.with_fallbacks(
            [parser_none], exception_key="parsing_error"
        )
        return RunnableMap(raw=self) | parser_with_fallback


def _chat_result(reply: str) -> ChatResult:
    """Wrap a reply in the result type of ``_generate``."""
    message = AIMessage(content=reply, response_metadata={"model_name": FAKE_MODEL_NAME})
    return ChatResult(generations=[ChatGeneration(message=message)])


def _reply_chunks(reply: str) -> Iterator[ChatGenerationChunk]:
    """Split a reply into word-sized chunks whose contents add up to the reply."""
    tokens = _TOKEN_PATTERN.findall(reply) or [""]
    last = len(tokens) - 1
    for index, token in enumerate(tokens):
        if index < last:
            yield ChatGenerationChunk(message=AIMessageChunk(content=token))
        else:
            # Metadata only on the last chunk: string values are concatenated when chunks merge.
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content=token,
                    chunk_position="last",
                    response_metadata={"model_name": FAKE_MODEL_NAME},
                )
            )


def _parse_structured_reply[ModelT: BaseModel](
    message: BaseMessage, schema: type[ModelT]
) -> ModelT:
    """Validate the reply text of ``message`` as JSON against ``schema``."""
    text = str(message.text)
    try:
        return schema.model_validate_json(text)
    except ValidationError as exc:
        msg = (
            f"The scripted reply is not valid JSON for {schema.__name__}. Reply: {text!r}. "
            "Script a FakeRule whose reply is a JSON object matching the schema. "
            f"Details: {exc}"
        )
        raise OutputParserException(msg, llm_output=text) from exc


def get_chat_model(settings: Settings) -> BaseChatModel:
    """Create the chat model selected by ``settings.llm_provider``.

    Args:
        settings: The settings to use. Library code receives them from its caller; only the
            entry points call ``get_settings()``.

    Returns:
        A ``ChatOllama`` for the ``ollama`` provider, or a ``ScriptedChatModel`` with
        ``DEFAULT_FAKE_RULES`` for the ``fake`` provider. The ``ChatOllama`` sends
        ``settings.ollama_num_ctx`` as the context window of every request, and its HTTP
        clients time out after ``settings.ollama_timeout_s`` seconds; no connection is made
        until the first call.

    Raises:
        ValueError: If the provider is not supported.
    """
    if settings.llm_provider == "ollama":
        # Imported here so that fake mode never loads the Ollama client.
        from langchain_ollama import ChatOllama

        logger.debug("Using Ollama model %r at %s", settings.ollama_model, settings.ollama_base_url)
        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=settings.llm_temperature,
            # Without num_ctx, the server's default context length applies (a few thousand
            # tokens on a small GPU), and Ollama silently truncates longer prompts.
            num_ctx=settings.ollama_num_ctx,
            # Passed to both the sync and the async httpx client; without it they never time
            # out. LangGraph cannot time out a sync node, so this is the only bound.
            client_kwargs={"timeout": settings.ollama_timeout_s},
        )
    if settings.llm_provider == "fake":
        logger.debug("Using the scripted fake chat model; no LLM is called")
        return ScriptedChatModel(rules=list(DEFAULT_FAKE_RULES), default_reply=DEFAULT_FAKE_REPLY)
    msg = f"Unsupported LLM provider: {settings.llm_provider!r}"
    raise ValueError(msg)
