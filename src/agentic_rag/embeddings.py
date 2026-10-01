"""Embedding model factory: a local sentence-transformers model, or an offline fake.

``get_embeddings(settings)`` is the only place that creates embedding models (plan decision
5), so indexing and querying always use the provider chosen by ``EMBEDDING_PROVIDER``:

- ``huggingface``: ``langchain_huggingface.HuggingFaceEmbeddings``, which runs the
  sentence-transformers model ``EMBEDDING_MODEL`` in this process, on the CPU, and returns
  L2-normalized vectors. The first use downloads the model into the Hugging Face cache
  (``HF_HOME``; a named volume in the container), later runs load it from there. The CPU is
  deliberate: torch comes from the CPU-only wheel index, and the GPU is left to Ollama.
- ``fake``: :class:`HashingEmbeddings`, a deterministic hashed bag of words. No model, no
  download, no torch. Texts that share words get similar vectors, so retrieval in fake mode
  still ranks lexically related chunks first.

Query and passage prompts: some model families were trained with fixed text prefixes and
retrieve worse without them. :func:`embedding_prompts` returns the prefixes for a model name;
for E5 models they are ``"query: "`` for queries and ``"passage: "`` for indexed text. They
reach sentence-transformers as the ``prompt`` argument of ``SentenceTransformer.encode``,
through ``encode_kwargs`` (used by ``embed_documents``) and ``query_encode_kwargs`` (used by
``embed_query``).

Consistency between indexing and querying: the model and its prompts depend only on
``EMBEDDING_PROVIDER`` and ``EMBEDDING_MODEL``, so the same settings always give the same
embedding model for building the index and for querying it. Vectors of different models are
not comparable: after changing either variable, rebuild the index
(``agentic-rag ingest --rebuild``).

Cost: creating the ``huggingface`` embeddings imports torch and loads the model, which takes
seconds and several hundred MB of memory even for a small model. Create the embeddings once
and reuse them: the RAG subgraph, for example, loads them once per compiled graph, through the
lazy index provider described in ``agentic_rag.rag.graph``. The heavy libraries are imported
only inside the ``huggingface`` branch, so importing this module and using the fake stay fast.
"""

import functools
import hashlib
import logging
import math
import re
import unicodedata
from collections import Counter
from typing import Any, Final, override

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict, Field

from agentic_rag.config import Settings

__all__ = [
    "E5_PASSAGE_PROMPT",
    "E5_QUERY_PROMPT",
    "FAKE_EMBEDDING_DIMENSION",
    "HUGGINGFACE_DEVICE",
    "EmbeddingPrompts",
    "HashingEmbeddings",
    "embedding_prompts",
    "get_embeddings",
]

logger = logging.getLogger(__name__)

HUGGINGFACE_DEVICE: Final = "cpu"
"""Torch device of the Hugging Face model: torch is the CPU-only build, the GPU is Ollama's."""

FAKE_EMBEDDING_DIMENSION: Final = 1024
"""Vector size of :class:`HashingEmbeddings`.

Large enough to keep hash collisions rare for chunk-sized texts, and different from the 384
dimensions of the default model (``intfloat/multilingual-e5-small``), so Chroma rejects
queries against an index that was built with the other provider instead of returning
meaningless matches.
"""

E5_QUERY_PROMPT: Final = "query: "
"""Prefix of search queries for E5 models (see the model cards of ``intfloat/*e5*``)."""

E5_PASSAGE_PROMPT: Final = "passage: "
"""Prefix of indexed passages for E5 models."""

# Model names are matched by their parts between these separators, so "e5" is recognised in
# "intfloat/multilingual-e5-small" or "C:\models\e5-base-v2" but not inside another word.
_NAME_SEPARATORS: Final = re.compile(r"[\s/\\_.:-]+")

# Word tokens of the fake embeddings: runs of letters, digits and underscores in any script.
_WORD_PATTERN: Final = re.compile(r"\w+")

# Stand-in token for texts without any word; the word pattern can never produce it.
_NO_WORDS_TOKEN: Final = "\x00"


class EmbeddingPrompts(BaseModel):
    """Text prefixes that an embedding model expects in front of queries and passages.

    Attributes:
        query: Prefix of search queries (``embed_query``); None when the model needs none.
        document: Prefix of indexed text (``embed_documents``); None when the model needs none.
    """

    model_config = ConfigDict(frozen=True)

    query: str | None = Field(default=None, description="Prefix of search queries.")
    document: str | None = Field(default=None, description="Prefix of indexed text.")


def embedding_prompts(model_name: str) -> EmbeddingPrompts:
    r"""Return the query and passage prefixes that a model expects.

    The model name is compared by its parts between separators (``/``, ``\``, ``-``, ``_``,
    ``.``, ``:`` and whitespace), ignoring letter case, so local model directories and
    fine-tuned derivatives are recognised as well:

    - E5 models (a part equal to ``e5``, for example ``intfloat/multilingual-e5-small``):
      :data:`E5_QUERY_PROMPT` and :data:`E5_PASSAGE_PROMPT`.
    - Any other model: no prefixes.

    Instruction-tuned E5 models (for example ``intfloat/multilingual-e5-large-instruct``)
    expect a task instruction instead of these prefixes and are not supported.

    Args:
        model_name: A Hugging Face model id or a local model directory.

    Returns:
        The prefixes; both are None for models that need none.
    """
    parts = set(_NAME_SEPARATORS.split(model_name.casefold()))
    if "e5" not in parts:
        return EmbeddingPrompts()
    return EmbeddingPrompts(query=E5_QUERY_PROMPT, document=E5_PASSAGE_PROMPT)


class HashingEmbeddings(BaseModel, Embeddings):
    """Deterministic, offline embeddings: a hashed bag of words with unit length.

    The fake embedding provider (``EMBEDDING_PROVIDER=fake``). A text becomes a vector of
    ``dimension`` floats in three steps:

    1. Normalize the text (Unicode NFKC, then case folding) and split it into words: runs of
       letters, digits and underscores, in any script.
    2. Hash every distinct word with BLAKE2b to one of the ``dimension`` positions and add its
       weight there: ``1 + ln(count)``, so repeated words count, but sublinearly.
    3. Divide the vector by its Euclidean length.

    The vectors depend only on the text and ``dimension``: they are the same in every process,
    on every platform and in every Python version (the built-in ``hash()`` is salted per
    process). All weights are positive, so the cosine similarity of two texts is positive when
    they share words and zero when they do not (unless two words collide in a position), and
    word order, letter case and punctuation make no difference. A text without any word (empty,
    whitespace or punctuation only) gets a fixed vector of its own, so every vector has length
    1. Queries and documents are embedded the same way.

    Only the synchronous methods are implemented: the graph nodes and the ingestion pipeline
    are synchronous. The asynchronous methods inherited from ``Embeddings`` run them in a
    thread pool.

    Example:
        >>> embeddings = HashingEmbeddings(dimension=64)
        >>> len(embeddings.embed_query("How do I renew my passport?"))
        64

    Attributes:
        dimension: Vector size.
    """

    model_config = ConfigDict(frozen=True)

    dimension: int = Field(
        default=FAKE_EMBEDDING_DIMENSION, ge=1, description="Vector size (number of positions)."
    )

    @override
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed texts for indexing.

        Args:
            texts: The texts to embed.

        Returns:
            One unit vector per text, in input order.
        """
        return [self._embed(text) for text in texts]

    @override
    def embed_query(self, text: str) -> list[float]:
        """Embed a search query, exactly like a document with the same text.

        Args:
            text: The query.

        Returns:
            A unit vector.
        """
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        """Embed one text with the method described in the class docstring."""
        counts = Counter(_words(text)) or Counter((_NO_WORDS_TOKEN,))
        vector = [0.0] * self.dimension
        for word, count in counts.items():
            vector[_position(word, self.dimension)] += 1.0 + math.log(count)
        # Every weight is at least 1, so the length is never zero.
        length = math.sqrt(math.fsum(value * value for value in vector))
        return [value / length for value in vector]


def get_embeddings(settings: Settings) -> Embeddings:
    """Create the embedding model selected by ``settings.embedding_provider``.

    The same settings always give the same model with the same prompts, so passages are
    indexed and queries are embedded consistently. Every call creates a new instance; the
    ``huggingface`` model takes seconds to load, so create it once and reuse it.

    Args:
        settings: The settings; only ``embedding_provider`` and ``embedding_model`` are used.

    Returns:
        For ``huggingface``, a ``HuggingFaceEmbeddings`` that runs ``settings.embedding_model``
        on the CPU with L2-normalized output and the prefixes of :func:`embedding_prompts`;
        the model is loaded now and downloaded into the Hugging Face cache on first use. For
        ``fake``, a :class:`HashingEmbeddings` with :data:`FAKE_EMBEDDING_DIMENSION`
        dimensions; ``settings.embedding_model`` is ignored.

    Raises:
        ValueError: If the provider is not supported. Errors of sentence-transformers and
            huggingface_hub propagate unchanged, for example when the model is neither in
            the cache nor downloadable.
    """
    if settings.embedding_provider == "huggingface":
        return _huggingface_embeddings(settings.embedding_model)
    if settings.embedding_provider == "fake":
        logger.debug("Using the offline hashing embeddings; no embedding model is loaded")
        return HashingEmbeddings()
    msg = f"Unsupported embedding provider: {settings.embedding_provider!r}"
    raise ValueError(msg)


def _huggingface_embeddings(model_name: str) -> Embeddings:
    """Create the sentence-transformers embeddings of ``model_name``; this loads the model."""
    # Imported here: creating the model imports sentence-transformers and torch, which takes
    # seconds and is never needed in fake mode.
    from langchain_huggingface import HuggingFaceEmbeddings

    prompts = embedding_prompts(model_name)
    logger.info(
        "Loading embedding model %r on the %s (the first use downloads it)",
        model_name,
        HUGGINGFACE_DEVICE,
    )
    # embed_query uses query_encode_kwargs instead of encode_kwargs, not on top of them, so
    # both carry the normalization. multi_process stays off: in langchain-huggingface 1.2 that
    # path ignores encode_kwargs, and with them the prompts and the normalization.
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": HUGGINGFACE_DEVICE},
        encode_kwargs=_encode_kwargs(prompts.document),
        query_encode_kwargs=_encode_kwargs(prompts.query),
    )


def _encode_kwargs(prompt: str | None) -> dict[str, Any]:
    """Arguments of ``SentenceTransformer.encode``: unit-length vectors and the prompt, if any."""
    kwargs: dict[str, Any] = {"normalize_embeddings": True}
    if prompt is not None:
        kwargs["prompt"] = prompt
    return kwargs


def _words(text: str) -> list[str]:
    """Split a text into normalized, case-folded words."""
    return _WORD_PATTERN.findall(unicodedata.normalize("NFKC", text).casefold())


@functools.lru_cache(maxsize=1 << 16)
def _position(word: str, dimension: int) -> int:
    """Map a word to a vector position with a stable hash (cached: corpora repeat words)."""
    digest = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dimension
