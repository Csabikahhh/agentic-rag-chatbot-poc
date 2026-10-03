"""Tests for agentic_rag.llm: provider selection and the scripted fake chat model."""

import json
import os
import socket
import threading
import time
import urllib.parse
import urllib.request
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, ValidationError

from agentic_rag import llm
from agentic_rag.config import Settings
from agentic_rag.llm import (
    DEFAULT_FAKE_REPLY,
    DEFAULT_FAKE_RULES,
    FAKE_MODEL_NAME,
    FakeRule,
    ScriptedChatModel,
    get_chat_model,
    render_prompt,
)

# Captured at import time, before conftest.py isolates the environment of each test, so the
# live test below can follow a developer's OLLAMA_BASE_URL and OLLAMA_MODEL.
LIVE_OLLAMA_URL = (
    os.environ.get("OLLAMA_BASE_URL") or Settings.model_fields["ollama_base_url"].default
)
LIVE_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL") or Settings.model_fields["ollama_model"].default


class Intent(BaseModel):
    """A small schema standing in for the structured outputs of later phases."""

    intent: str
    confidence: float


def scripted(*rules: tuple[str, str], **kwargs: Any) -> ScriptedChatModel:
    """Build a fake model from (pattern, reply) pairs."""
    return ScriptedChatModel(
        rules=[FakeRule(pattern=pattern, reply=reply) for pattern, reply in rules], **kwargs
    )


def ollama_settings(base_url: str, **overrides: Any) -> Settings:
    """Settings for the ``ollama`` provider against a stand-in server at ``base_url``."""
    return Settings(
        _env_file=None,
        llm_provider="ollama",
        ollama_base_url=base_url,
        ollama_model="tiny-model:1b",
        **overrides,
    )


# --- provider selection -------------------------------------------------------------------


def test_fake_provider_gives_the_scripted_model(settings: Settings) -> None:
    model = get_chat_model(settings)

    assert isinstance(model, ScriptedChatModel)
    assert model.rules == list(DEFAULT_FAKE_RULES)
    assert model.default_reply == DEFAULT_FAKE_REPLY


def test_ollama_provider_gives_chat_ollama_without_connecting() -> None:
    # Port 9 (discard) has no Ollama behind it: construction must not need a server.
    settings = ollama_settings(
        "http://127.0.0.1:9", llm_temperature=0.3, ollama_num_ctx=4096, ollama_timeout_s=45.0
    )

    model = get_chat_model(settings)

    assert isinstance(model, ChatOllama)
    assert (model.model, model.base_url, model.temperature, model.num_ctx) == (
        "tiny-model:1b",
        "http://127.0.0.1:9",
        0.3,
        4096,
    )
    assert model.client_kwargs == {"timeout": 45.0}
    assert model.reasoning is None  # The model's default thinking mode.


def test_get_chat_model_requires_explicit_settings() -> None:
    # Only the entry points read the process settings (get_settings); library code passes them.
    with pytest.raises(TypeError):
        get_chat_model()  # type: ignore[call-arg]


def test_fake_provider_uses_the_current_default_rules(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(llm, "DEFAULT_FAKE_RULES", (FakeRule(pattern="ping", reply="pong"),))

    assert get_chat_model(settings).invoke("ping").content == "pong"


def test_default_fake_rules_are_an_immutable_tuple_of_rules() -> None:
    # A tuple, so no test or module can append to the rules of fake mode by accident.
    assert isinstance(DEFAULT_FAKE_RULES, tuple)
    assert all(isinstance(rule, FakeRule) for rule in DEFAULT_FAKE_RULES)


# --- Ollama requests (local stand-in servers, no Ollama needed) ---------------------------


@pytest.fixture
def ollama_stand_in() -> Iterator[tuple[str, list[dict[str, Any]]]]:
    """A local HTTP server that answers ``/api/chat`` like Ollama.

    Yields:
        The server's base URL and the list that collects the JSON body of every request.
    """
    received: list[dict[str, Any]] = []
    answer = {
        "model": "tiny-model:1b",
        "created_at": "2026-01-01T00:00:00Z",
        "message": {"role": "assistant", "content": "OK"},
        "done": True,
        "done_reason": "stop",
    }

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            payload = (json.dumps(answer) + "\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: Any) -> None:
            """Keep the request log out of the test output."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", received
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def stalled_server() -> Iterator[str]:
    """A local port that accepts connections but never answers, like a stalled Ollama."""
    with socket.create_server(("127.0.0.1", 0)) as server:
        yield f"http://127.0.0.1:{server.getsockname()[1]}"


def test_ollama_requests_send_the_context_window(
    ollama_stand_in: tuple[str, list[dict[str, Any]]],
) -> None:
    base_url, received = ollama_stand_in
    model = get_chat_model(ollama_settings(base_url, ollama_num_ctx=4096))

    reply = model.invoke("Reply with the single word OK.")

    assert reply.content == "OK"
    assert [body["model"] for body in received] == ["tiny-model:1b"]
    # Without num_ctx in the options, Ollama would apply its own default context length.
    assert received[0]["options"]["num_ctx"] == 4096
    assert "think" not in received[0]  # OLLAMA_REASONING unset: the model's default.


def test_ollama_requests_can_turn_thinking_off(
    ollama_stand_in: tuple[str, list[dict[str, Any]]],
) -> None:
    base_url, received = ollama_stand_in
    model = get_chat_model(ollama_settings(base_url, ollama_reasoning=False))

    model.invoke("Reply with the single word OK.")

    assert received[0]["think"] is False


def test_ollama_requests_time_out_after_the_configured_seconds(stalled_server: str) -> None:
    model = get_chat_model(ollama_settings(stalled_server, ollama_timeout_s=0.3))

    started = time.perf_counter()
    with pytest.raises(httpx.TimeoutException):
        model.invoke("Is anybody there?")

    assert time.perf_counter() - started < 10


# --- rules and default reply --------------------------------------------------------------


def test_first_matching_rule_wins() -> None:
    model = scripted(("capital of (France|Italy)", "first"), ("France", "second"))

    assert model.invoke("What is the capital of France?").content == "first"
    assert model.invoke("France or Spain?").content == "second"
    assert model.invoke("Unrelated question").content == DEFAULT_FAKE_REPLY


def test_patterns_are_searched_and_take_inline_flags() -> None:
    model = scripted(("(?i)^summarize", "summary"), (r"\bdeadline\b", "date"))

    assert model.invoke("SUMMARIZE the text").content == "summary"
    assert model.invoke("When is the deadline?").content == "date"
    assert model.invoke("deadlines").content == DEFAULT_FAKE_REPLY


def test_rules_see_every_message_of_the_prompt() -> None:
    model = scripted(("(?is)ROUTER.*compare", '{"intent": "complex"}'))
    messages = [SystemMessage("ROUTER: classify the request."), HumanMessage("Compare A and B")]

    assert model.invoke(messages).content == '{"intent": "complex"}'
    assert model.last_prompt == "ROUTER: classify the request.\n\nCompare A and B"
    assert render_prompt(messages) == model.last_prompt


def test_an_expanding_rule_fills_in_the_groups_of_its_match() -> None:
    rule = FakeRule(
        pattern=r"contrast of (?P<fg>#\w+) on (?P<bg>#\w+)",
        reply=r'{"foreground": "\g<fg>", "background": "\2"}',
        expand=True,
    )
    model = ScriptedChatModel(rules=[rule])

    assert model.invoke("the contrast of #777 on #fff").content == (
        '{"foreground": "#777", "background": "#fff"}'
    )


def test_a_plain_rule_keeps_its_reply_literally() -> None:
    model = scripted((r"(?P<word>\w+)", r"\g<word>"))

    assert model.invoke("anything").content == r"\g<word>"


def test_default_reply_is_deterministic_and_configurable() -> None:
    assert ScriptedChatModel().invoke("a").content == DEFAULT_FAKE_REPLY
    assert ScriptedChatModel().invoke("b").content == DEFAULT_FAKE_REPLY
    assert ScriptedChatModel(default_reply="no idea").invoke("anything").content == "no idea"


def test_reply_for_does_not_record_the_prompt() -> None:
    model = scripted(("ping", "pong"))

    assert model.reply_for("ping") == "pong"
    assert model.prompts == ()


def test_invalid_rule_pattern_is_rejected_when_defined() -> None:
    with pytest.raises(ValidationError, match="invalid regular expression"):
        FakeRule(pattern="(unclosed", reply="never")


def test_replies_carry_the_fake_model_name() -> None:
    reply = ScriptedChatModel().invoke("Hello")

    assert isinstance(reply, AIMessage)
    assert reply.response_metadata["model_name"] == FAKE_MODEL_NAME


# --- prompt recording ---------------------------------------------------------------------


def test_prompts_are_recorded_in_order_as_immutable_snapshots() -> None:
    model = ScriptedChatModel()
    assert (model.prompts, model.last_prompt) == ((), None)

    model.invoke("first")
    model.invoke([HumanMessage("second")])
    snapshot = model.prompts
    model.invoke("third")

    assert snapshot == ("first", "second")
    assert model.prompts == ("first", "second", "third")
    assert model.last_prompt == "third"

    model.clear_prompts()
    assert model.prompts == ()


def test_batch_records_every_prompt() -> None:
    model = scripted(("one", "1"), ("two", "2"))

    replies = model.batch(["one", "two"])

    assert [reply.content for reply in replies] == ["1", "2"]
    assert sorted(model.prompts) == ["one", "two"]


# --- streaming ----------------------------------------------------------------------------


def test_stream_yields_chunks_that_add_up_to_the_reply() -> None:
    model = scripted(("story", "Once upon  a time.\n"))

    chunks = list(model.stream("Tell a story"))

    assert all(isinstance(chunk, AIMessageChunk) for chunk in chunks)
    assert len(chunks) > 1
    assert "".join(str(chunk.content) for chunk in chunks) == "Once upon  a time.\n"
    assert model.prompts == ("Tell a story",)


def test_empty_reply_streams_as_one_empty_chunk() -> None:
    chunks = list(ScriptedChatModel(default_reply="").stream("anything"))

    assert [chunk.content for chunk in chunks] == [""]


# --- structured output --------------------------------------------------------------------


def test_with_structured_output_returns_a_validated_model() -> None:
    model = scripted(("classify", '{"intent": "single", "confidence": 0.9}'))

    result = model.with_structured_output(Intent).invoke("classify this")

    assert result == Intent(intent="single", confidence=0.9)
    assert model.prompts == ("classify this",)


def test_with_structured_output_ignores_the_ollama_method_and_strict_arguments() -> None:
    model = scripted(("classify", '{"intent": "tool", "confidence": 1}'))
    structured = model.with_structured_output(Intent, method="json_schema", strict=True)

    assert structured.invoke("classify this") == Intent(intent="tool", confidence=1.0)


def test_with_structured_output_rejects_a_reply_that_is_not_json() -> None:
    structured = ScriptedChatModel().with_structured_output(Intent)

    with pytest.raises(OutputParserException, match="not valid JSON for Intent") as excinfo:
        structured.invoke("classify this")

    assert excinfo.value.llm_output == DEFAULT_FAKE_REPLY


def test_with_structured_output_rejects_json_of_the_wrong_shape() -> None:
    structured = scripted(("classify", '{"intent": "single"}')).with_structured_output(Intent)

    with pytest.raises(OutputParserException, match="confidence"):
        structured.invoke("classify this")


def test_with_structured_output_include_raw_reports_instead_of_raising() -> None:
    model = scripted(("good", '{"intent": "direct", "confidence": 0.5}'), ("bad", "nope"))
    structured = model.with_structured_output(Intent, include_raw=True)

    good = structured.invoke("good")
    bad = structured.invoke("bad")

    assert good["parsed"] == Intent(intent="direct", confidence=0.5)
    assert good["parsing_error"] is None
    assert isinstance(good["raw"], AIMessage)
    assert bad["parsed"] is None
    assert isinstance(bad["parsing_error"], OutputParserException)
    assert bad["raw"].content == "nope"


def test_with_structured_output_needs_a_pydantic_model_class() -> None:
    model = ScriptedChatModel()

    with pytest.raises(TypeError, match="Pydantic model classes only"):
        model.with_structured_output({"type": "object", "properties": {}})
    with pytest.raises(ValueError, match="unsupported arguments"):
        model.with_structured_output(Intent, temperature=0)


# --- live check (deselected by default; run with: uv run pytest -m ollama) ----------------


def _ollama_serves(base_url: str, model: str) -> bool:
    """Return whether an Ollama server answers at ``base_url`` and has ``model`` pulled."""
    url = urllib.parse.urlsplit(base_url)
    address = (url.hostname or "localhost", url.port or (443 if url.scheme == "https" else 80))
    try:
        # A quick TCP check first: on Windows a closed port takes seconds to refuse a connection.
        with socket.create_connection(address, timeout=0.25):
            pass
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=5) as response:
            payload = json.load(response)
    except (OSError, ValueError):
        return False
    names = {entry.get("name") for entry in payload.get("models", [])}
    return model in names or f"{model}:latest" in names


@pytest.mark.ollama
def test_live_ollama_answers_a_trivial_prompt() -> None:
    if not _ollama_serves(LIVE_OLLAMA_URL, LIVE_OLLAMA_MODEL):
        pytest.skip(f"Ollama with {LIVE_OLLAMA_MODEL!r} is not reachable at {LIVE_OLLAMA_URL}")
    settings = Settings(
        _env_file=None,
        llm_provider="ollama",
        ollama_base_url=LIVE_OLLAMA_URL,
        ollama_model=LIVE_OLLAMA_MODEL,
    )

    reply = get_chat_model(settings).invoke("Reply with the single word OK.")

    assert str(reply.content).strip()
