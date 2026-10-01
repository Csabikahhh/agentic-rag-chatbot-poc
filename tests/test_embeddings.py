"""Tests for agentic_rag.embeddings: the offline fake provider and the Hugging Face branch.

No test loads a real model. The Hugging Face branch is checked twice: with a stand-in for the
``HuggingFaceEmbeddings`` class (the constructor arguments), and with the real
langchain-huggingface class running on a stand-in ``sentence_transformers`` module (the
arguments that reach ``SentenceTransformer.encode``). Neither imports torch or downloads
anything.
"""

import inspect
import json
import math
import os
import subprocess
import sys
import types
import unicodedata
from typing import Any

import numpy as np
import pytest
from langchain_core.embeddings import Embeddings
from pydantic import ValidationError

from agentic_rag.config import Settings
from agentic_rag.embeddings import (
    E5_PASSAGE_PROMPT,
    E5_QUERY_PROMPT,
    FAKE_EMBEDDING_DIMENSION,
    HUGGINGFACE_DEVICE,
    EmbeddingPrompts,
    HashingEmbeddings,
    embedding_prompts,
    get_embeddings,
)

DEFAULT_MODEL = "intfloat/multilingual-e5-small"
SAMPLE_TEXTS = (
    "How do I renew my passport?",
    "Árvíztűrő tükörfúrógép",
    "",
    "   ?!   ",
    "repeated repeated repeated word",
    "A longer passage. " * 60,
)

E5_PROMPTS = EmbeddingPrompts(query=E5_QUERY_PROMPT, document=E5_PASSAGE_PROMPT)
NO_PROMPTS = EmbeddingPrompts()

# Runs in a fresh interpreter: embeds argv[1] (a JSON list of texts) with the fake provider.
# That this loads no model library is checked in test_imports.py.
_FAKE_PROVIDER_PROBE = """
import json, sys
from agentic_rag.config import Settings
from agentic_rag.embeddings import get_embeddings
embeddings = get_embeddings(Settings(_env_file=None, embedding_provider="fake"))
vectors = embeddings.embed_documents(json.loads(sys.argv[1]))
print(json.dumps(vectors))
"""


def with_changes(settings: Settings, **changes: Any) -> Settings:
    """A validated copy of ``settings`` with ``changes`` applied; the environment is not read."""
    return Settings.model_validate({**settings.model_dump(), **changes})


def length(vector: list[float]) -> float:
    """Euclidean length of a vector."""
    return math.sqrt(math.fsum(value * value for value in vector))


def cosine(first: list[float], second: list[float]) -> float:
    """Cosine similarity of two vectors."""
    dot = math.fsum(a * b for a, b in zip(first, second, strict=True))
    return dot / (length(first) * length(second))


class RecordingEmbeddings:
    """Stand-in for ``HuggingFaceEmbeddings``: records its constructor arguments, loads nothing."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


@pytest.fixture
def recording_class(monkeypatch: pytest.MonkeyPatch) -> type[RecordingEmbeddings]:
    """Replace ``langchain_huggingface.HuggingFaceEmbeddings`` with ``RecordingEmbeddings``."""
    import langchain_huggingface

    monkeypatch.setattr(langchain_huggingface, "HuggingFaceEmbeddings", RecordingEmbeddings)
    return RecordingEmbeddings


@pytest.fixture
def hf_settings(settings: Settings) -> Settings:
    """The offline test settings, switched to the Hugging Face provider."""
    return with_changes(settings, embedding_provider="huggingface")


@pytest.fixture
def sentence_transformer_instances(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Install a stand-in ``sentence_transformers`` module for one test.

    Returns:
        The stand-in models created during the test. Each records its constructor arguments
        and every ``encode`` call, and returns 3-dimensional vectors.
    """
    instances: list[Any] = []

    class StubSentenceTransformer:
        """Stand-in for ``sentence_transformers.SentenceTransformer``."""

        def __init__(self, model_name_or_path: str, **kwargs: Any) -> None:
            self.model_name_or_path = model_name_or_path
            self.init_kwargs = kwargs
            self.encode_calls: list[tuple[list[str], dict[str, Any]]] = []
            instances.append(self)

        def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
            self.encode_calls.append((list(texts), kwargs))
            return np.full((len(texts), 3), 0.5)

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = StubSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return instances


@pytest.fixture(scope="module")
def fake_provider_vectors(tmp_path_factory: pytest.TempPathFactory) -> list[list[float]]:
    """Embed ``SAMPLE_TEXTS`` with the fake provider in a fresh interpreter.

    The child process gets its own hash seed and none of the settings variables of this
    process.

    Returns:
        The vectors of ``SAMPLE_TEXTS``, computed by the child process.
    """
    field_names = set(Settings.model_fields)
    env = {name: value for name, value in os.environ.items() if name.lower() not in field_names}
    env["PYTHONHASHSEED"] = "1234"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _FAKE_PROVIDER_PROBE,
            json.dumps(SAMPLE_TEXTS),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=tmp_path_factory.mktemp("fake-provider-probe"),
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


# --- the fake provider ----------------------------------------------------------------------


def test_fake_provider_gives_hashing_embeddings(settings: Settings) -> None:
    embeddings = get_embeddings(settings)

    assert isinstance(embeddings, HashingEmbeddings)
    assert isinstance(embeddings, Embeddings)
    assert embeddings.dimension == FAKE_EMBEDDING_DIMENSION


def test_get_embeddings_requires_explicit_settings() -> None:
    # No default: the factory never falls back to the process-wide get_settings().
    parameter = inspect.signature(get_embeddings).parameters["settings"]

    assert parameter.default is inspect.Parameter.empty


def test_fake_provider_ignores_the_embedding_model(settings: Settings) -> None:
    first = get_embeddings(with_changes(settings, embedding_model=DEFAULT_MODEL))
    second = get_embeddings(with_changes(settings, embedding_model="other/model"))

    assert first.embed_query("same text") == second.embed_query("same text")


@pytest.mark.parametrize("dimension", [1, 16, FAKE_EMBEDDING_DIMENSION])
def test_vectors_have_the_configured_dimension(dimension: int) -> None:
    embeddings = HashingEmbeddings(dimension=dimension)

    vectors = [*embeddings.embed_documents(list(SAMPLE_TEXTS)), embeddings.embed_query("q")]

    assert {len(vector) for vector in vectors} == {dimension}


def test_fake_dimension_differs_from_the_default_model() -> None:
    # multilingual-e5-small returns 384 dimensions; a different size makes Chroma reject an
    # index built with the other provider instead of returning meaningless matches.
    assert FAKE_EMBEDDING_DIMENSION != 384


@pytest.mark.parametrize("dimension", [0, -1])
def test_invalid_dimension_is_rejected(dimension: int) -> None:
    with pytest.raises(ValidationError):
        HashingEmbeddings(dimension=dimension)


def test_hashing_embeddings_are_immutable() -> None:
    embeddings = HashingEmbeddings()

    with pytest.raises(ValidationError):
        embeddings.dimension = 8  # type: ignore[misc]


@pytest.mark.parametrize("text", SAMPLE_TEXTS)
def test_vectors_have_unit_length(text: str) -> None:
    assert length(HashingEmbeddings().embed_query(text)) == pytest.approx(1.0)


def test_same_text_gives_the_same_vector() -> None:
    first, second = HashingEmbeddings(), HashingEmbeddings()

    assert first.embed_query("solar panel") == first.embed_query("solar panel")
    assert first.embed_query("solar panel") == second.embed_query("solar panel")


def test_vectors_do_not_depend_on_the_process(fake_provider_vectors: list[list[float]]) -> None:
    expected = HashingEmbeddings().embed_documents(list(SAMPLE_TEXTS))

    assert fake_provider_vectors == expected


@pytest.mark.parametrize("text", SAMPLE_TEXTS)
def test_embed_query_equals_embed_documents(text: str) -> None:
    embeddings = HashingEmbeddings()

    assert embeddings.embed_query(text) == embeddings.embed_documents([text])[0]


def test_embed_documents_keeps_order_and_length() -> None:
    embeddings = HashingEmbeddings()
    texts = ["alpha beta", "gamma", "delta epsilon"]

    vectors = embeddings.embed_documents(texts)

    assert vectors == [embeddings.embed_query(text) for text in texts]
    assert embeddings.embed_documents([]) == []


def test_texts_sharing_more_words_rank_higher() -> None:
    embeddings = HashingEmbeddings()
    documents = [
        "Solar panel installation guide",
        "Solar energy basics",
        "Tax return deadlines",
    ]

    query = embeddings.embed_query("How does solar panel installation work?")
    scores = [cosine(query, vector) for vector in embeddings.embed_documents(documents)]

    assert scores[0] > scores[1] > scores[2]


def test_identical_texts_are_parallel_and_unrelated_texts_orthogonal() -> None:
    embeddings = HashingEmbeddings()
    query = embeddings.embed_query("solar panel installation")

    assert cosine(query, embeddings.embed_query("solar panel installation")) == pytest.approx(1)
    assert cosine(query, embeddings.embed_query("tax return deadlines")) == pytest.approx(0)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("Solar panel!", "panel, SOLAR"),
        ("Árvíztűrő tükörfúrógép", "ÁRVÍZTŰRŐ TÜKÖRFÚRÓGÉP"),
        (
            unicodedata.normalize("NFC", "árvíztűrő"),
            unicodedata.normalize("NFD", "árvíztűrő"),
        ),
    ],
    ids=["order-case-punctuation", "hungarian-case", "unicode-normal-forms"],
)
def test_order_case_punctuation_and_normal_form_do_not_matter(first: str, second: str) -> None:
    embeddings = HashingEmbeddings()

    assert embeddings.embed_query(first) == embeddings.embed_query(second)


def test_texts_without_words_share_one_unit_vector() -> None:
    embeddings = HashingEmbeddings()
    empty = embeddings.embed_query("")

    assert embeddings.embed_query("   ") == empty
    assert embeddings.embed_query("?! ...") == empty
    assert cosine(empty, embeddings.embed_query("word")) < 1


def test_repeated_words_weigh_more_but_sublinearly() -> None:
    embeddings = HashingEmbeddings()
    cat = embeddings.embed_query("cat")

    once = cosine(cat, embeddings.embed_query("cat dog"))
    thrice = cosine(cat, embeddings.embed_query("cat cat cat dog"))

    assert once < thrice < cosine(cat, embeddings.embed_query("cat"))
    # With linear counts "cat" would weigh 3; with 1 + ln(3) it weighs about 2.1.
    assert thrice == pytest.approx((1 + math.log(3)) / math.sqrt((1 + math.log(3)) ** 2 + 1))


# --- prompts --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        (DEFAULT_MODEL, E5_PROMPTS),
        ("intfloat/e5-base-v2", E5_PROMPTS),
        ("intfloat/E5-Large", E5_PROMPTS),
        ("someone/multilingual-e5-small_finetuned-hu", E5_PROMPTS),
        ("/models/multilingual-e5-small", E5_PROMPTS),
        ("C:\\models\\e5-small-v2", E5_PROMPTS),
        ("BAAI/bge-m3", NO_PROMPTS),
        ("sentence-transformers/all-MiniLM-L6-v2", NO_PROMPTS),
        ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", NO_PROMPTS),
        ("example/large5-encoder", NO_PROMPTS),
    ],
)
def test_embedding_prompts_recognise_e5_models(model_name: str, expected: EmbeddingPrompts) -> None:
    assert embedding_prompts(model_name) == expected


def test_e5_prompts_are_the_documented_prefixes() -> None:
    assert (E5_QUERY_PROMPT, E5_PASSAGE_PROMPT) == ("query: ", "passage: ")


# --- the Hugging Face provider --------------------------------------------------------------


def test_huggingface_provider_runs_e5_on_the_cpu_with_normalized_prompted_vectors(
    hf_settings: Settings, recording_class: type[RecordingEmbeddings]
) -> None:
    embeddings = get_embeddings(hf_settings)

    assert isinstance(embeddings, recording_class)
    assert embeddings.kwargs == {
        "model_name": DEFAULT_MODEL,
        "model_kwargs": {"device": "cpu"},
        "encode_kwargs": {"normalize_embeddings": True, "prompt": "passage: "},
        "query_encode_kwargs": {"normalize_embeddings": True, "prompt": "query: "},
    }
    assert HUGGINGFACE_DEVICE == "cpu"


def test_huggingface_provider_adds_no_prompts_for_other_models(
    hf_settings: Settings, recording_class: type[RecordingEmbeddings]
) -> None:
    model = "sentence-transformers/all-MiniLM-L6-v2"

    embeddings = get_embeddings(with_changes(hf_settings, embedding_model=model))

    assert isinstance(embeddings, recording_class)
    assert embeddings.kwargs["model_name"] == model
    assert embeddings.kwargs["encode_kwargs"] == {"normalize_embeddings": True}
    assert embeddings.kwargs["query_encode_kwargs"] == {"normalize_embeddings": True}


def test_huggingface_is_the_default_provider(recording_class: type[RecordingEmbeddings]) -> None:
    # No settings fixture: the isolated environment leaves every field at its default.
    embeddings = get_embeddings(Settings(_env_file=None))

    assert isinstance(embeddings, recording_class)
    assert embeddings.kwargs["model_name"] == DEFAULT_MODEL


def test_same_settings_give_the_same_model_for_indexing_and_querying(
    hf_settings: Settings, recording_class: type[RecordingEmbeddings]
) -> None:
    for_indexing = get_embeddings(hf_settings)
    for_querying = get_embeddings(hf_settings)

    assert isinstance(for_indexing, recording_class)
    assert isinstance(for_querying, recording_class)
    assert for_indexing.kwargs == for_querying.kwargs


def test_langchain_huggingface_passes_the_prompts_to_sentence_transformers(
    hf_settings: Settings, sentence_transformer_instances: list[Any]
) -> None:
    embeddings = get_embeddings(hf_settings)

    query_vector = embeddings.embed_query("Mikor kell beadni a bevallást?")
    document_vectors = embeddings.embed_documents(["First passage.", "Second passage."])

    (model,) = sentence_transformer_instances
    assert model.model_name_or_path == DEFAULT_MODEL
    assert model.init_kwargs["device"] == "cpu"
    (query_texts, query_kwargs), (document_texts, document_kwargs) = model.encode_calls
    assert query_texts == ["Mikor kell beadni a bevallást?"]
    assert query_kwargs["prompt"] == E5_QUERY_PROMPT
    assert query_kwargs["normalize_embeddings"] is True
    assert document_texts == ["First passage.", "Second passage."]
    assert document_kwargs["prompt"] == E5_PASSAGE_PROMPT
    assert document_kwargs["normalize_embeddings"] is True
    assert query_vector == [0.5, 0.5, 0.5]
    assert document_vectors == [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]


def test_unsupported_provider_is_rejected(settings: Settings) -> None:
    # Settings validation already rejects unknown providers; model_copy skips validation.
    broken = settings.model_copy(update={"embedding_provider": "openai"})

    with pytest.raises(ValueError, match="Unsupported embedding provider: 'openai'"):
        get_embeddings(broken)
