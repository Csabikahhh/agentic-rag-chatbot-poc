"""Persistent vector index: build and open the Chroma collection (plan section 5.4, decision 6).

The index is the Chroma collection ``settings.chroma_collection`` in a persistent client
directory, ``settings.chroma_dir`` (``data/chroma_db``: gitignored locally, a named volume in
the container):

- :func:`build_index` runs the pipeline: ``loaders.load_documents`` ->
  ``chunking.split_documents`` -> embed with ``get_embeddings(settings)`` -> add the chunks the
  collection does not hold yet, under each chunk's ``chunk_id`` -> delete the stored chunks
  that the run did not produce. It backs ``agentic-rag ingest`` and the start-up ingestion of
  the container (``INGEST_ON_START``).
- :func:`load_index` opens the existing collection for the ``retrieve`` node of the RAG
  subgraph.
- :func:`index_state` tells, without loading the embedding model, whether the collection is
  missing, ready for the current embedding settings, or built with other ones; the start-up
  preparation (``agentic_rag.ingestion.prepare``) rebuilds it in the last case.

Both functions take the embedding model from ``agentic_rag.embeddings.get_embeddings``, so the
same settings embed the passages and the queries with the same model. The collection records
``EMBEDDING_PROVIDER`` and ``EMBEDDING_MODEL`` in its metadata, so an index cannot be queried
with another model by mistake; for the ``fake`` provider, which ignores the model, only the
provider has to match. Both providers return unit-length vectors, so the collection uses
cosine distance and the relevance scores are ``1 - distance`` (higher is more relevant, as
``Source.score`` expects).

chromadb and langchain_chroma are imported inside the functions, so importing this module
stays fast. Chroma's anonymous telemetry is turned off.
"""

import logging
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, ConfigDict, Field

from agentic_rag.config import EmbeddingProvider, Settings
from agentic_rag.embeddings import get_embeddings
from agentic_rag.errors import EmbeddingMismatchError, IndexNotFoundError
from agentic_rag.ingestion.chunking import ChunkingConfig, split_documents
from agentic_rag.ingestion.loaders import list_corpus_files, load_documents

if TYPE_CHECKING:
    from chromadb.api import ClientAPI
    from chromadb.api.models.Collection import Collection

__all__ = [
    "COLLECTION_CONFIGURATION",
    "EMBEDDING_BATCH_SIZE",
    "EmbeddingMismatchError",
    "IndexNotFoundError",
    "IndexState",
    "IndexStats",
    "build_index",
    "index_state",
    "load_index",
]

type IndexState = Literal["missing", "ready", "mismatch"]
"""What :func:`index_state` finds: no collection (or an empty one), a collection built with
the current embedding settings, or one built with other embeddings."""

logger = logging.getLogger(__name__)

COLLECTION_CONFIGURATION: Final = {"hnsw": {"space": "cosine"}}
"""Index configuration of the collection: cosine distance, so scores are ``1 - distance``."""

EMBEDDING_BATCH_SIZE: Final = 256
"""Chunks embedded and written per batch; the progress is logged after every batch."""

_PROVIDER_KEY: Final = "embedding_provider"
_MODEL_KEY: Final = "embedding_model"


class IndexStats(BaseModel):
    """Outcome of one :func:`build_index` run; ``agentic-rag ingest`` prints it as JSON.

    Attributes:
        collection: Name of the Chroma collection.
        chroma_dir: Directory of the persistent Chroma client.
        embedding_provider: Embedding provider the vectors were computed with.
        embedding_model: Configured embedding model (not used by the ``fake`` provider).
        chunking: Splitter configuration the chunks were made with.
        files: Number of corpus files loaded.
        documents: Number of documents the loaders produced (for example one per PDF page).
        chunks: Number of chunks the run produced, which is also the number of chunks in
            the collection after the run.
        rebuilt: Whether the collection was deleted and built from scratch.
        duration_ms: Wall-clock duration of the run in milliseconds.
    """

    model_config = ConfigDict(frozen=True)

    collection: str = Field(min_length=1, description="Name of the Chroma collection.")
    chroma_dir: Path = Field(description="Directory of the persistent Chroma client.")
    embedding_provider: EmbeddingProvider = Field(
        description="Embedding provider the vectors were computed with."
    )
    embedding_model: str = Field(
        min_length=1, description="Configured embedding model (unused by the fake provider)."
    )
    chunking: ChunkingConfig = Field(
        default_factory=ChunkingConfig,
        description="Splitter configuration the chunks were made with.",
    )
    files: int = Field(ge=0, description="Number of corpus files loaded.")
    documents: int = Field(ge=0, description="Number of documents the loaders produced.")
    chunks: int = Field(
        ge=0, description="Number of chunks the run produced and the collection holds after it."
    )
    rebuilt: bool = Field(
        default=False, description="Whether the collection was built from scratch."
    )
    duration_ms: float = Field(ge=0, description="Wall-clock duration in milliseconds.")


def build_index(settings: Settings, *, rebuild: bool = False) -> IndexStats:
    """Build or update the vector index from the corpus: load, split, embed and store.

    Each run leaves the collection with exactly the chunks of the current corpus:

    1. Add every chunk the run produced that the collection does not hold yet, under its
       ``chunk_id``. A chunk id is a hash of the chunk's metadata, position and text, so a
       stored id already holds that very chunk, and only new or changed chunks are embedded:
       a run over an unchanged corpus embeds nothing and does not load the embedding model.
    2. Once the upsert has succeeded, delete every stored id that the run did not produce:
       the set difference of the stored ids and the produced ids.

    Both steps send batches of at most :data:`EMBEDDING_BATCH_SIZE` chunks (fewer than the
    client's maximum batch size). The set difference handles every change to the corpus:
    added, edited, shortened, re-chunked and removed files. Deletion comes last, so a run that
    fails without ``rebuild`` never removes chunks. An unchanged corpus leaves the collection
    as it was, so the command is idempotent.

    Args:
        settings: ``data_dir`` is the corpus; ``chroma_dir`` and ``chroma_collection`` are the
            target; ``embedding_provider`` and ``embedding_model`` select the embeddings.
        rebuild: Delete the collection first and build it from scratch. This is needed after
            changing the embedding provider or model; a plain run handles changes to the
            corpus and to the chunking.

    Returns:
        What the run loaded and stored. ``chunks`` is the number of chunks the run produced,
        which is also the size of the collection afterwards.

    Raises:
        FileNotFoundError: If ``settings.data_dir`` does not exist or holds no document with
            text. An empty corpus never empties an existing index.
        EmbeddingMismatchError: If the existing collection was built with another embedding
            provider or model and ``rebuild`` is False.
    """
    started = time.perf_counter()
    files = list_corpus_files(settings.data_dir)
    documents = load_documents(settings)
    if not documents:
        msg = (
            f"The corpus in {settings.data_dir} has no documents; run "
            "'agentic-rag ingest --download' to download it"
        )
        raise FileNotFoundError(msg)
    config = ChunkingConfig()
    chunks = split_documents(documents, config=config)

    client = _client(settings.chroma_dir)
    if rebuild and _collection_names(client) & {settings.chroma_collection}:
        logger.info("Deleting the collection %r to rebuild it", settings.chroma_collection)
        client.delete_collection(settings.chroma_collection)
    collection = client.get_or_create_collection(
        settings.chroma_collection,
        metadata={
            _PROVIDER_KEY: settings.embedding_provider,
            _MODEL_KEY: settings.embedding_model,
        },
        configuration=COLLECTION_CONFIGURATION,  # type: ignore[arg-type]
    )
    _check_embeddings(collection, settings)

    stored = set(collection.get(include=[])["ids"])
    produced = {chunk.metadata["chunk_id"] for chunk in chunks}
    new_chunks = [chunk for chunk in chunks if chunk.metadata["chunk_id"] not in stored]
    if new_chunks:
        _add_chunks(collection, new_chunks, settings)
    stale = sorted(stored - produced)
    for batch in _batches(stale, EMBEDDING_BATCH_SIZE):
        collection.delete(ids=batch)
    logger.info(
        "Index %r: %d chunks (%d added, %d removed, %d unchanged)",
        settings.chroma_collection,
        len(chunks),
        len(new_chunks),
        len(stale),
        len(chunks) - len(new_chunks),
    )
    return IndexStats(
        collection=settings.chroma_collection,
        chroma_dir=settings.chroma_dir,
        embedding_provider=settings.embedding_provider,
        embedding_model=settings.embedding_model,
        chunking=config,
        files=len(files),
        documents=len(documents),
        chunks=len(chunks),
        rebuilt=rebuild,
        duration_ms=(time.perf_counter() - started) * 1000,
    )


def load_index(settings: Settings) -> VectorStore:
    """Open the existing vector index for retrieval.

    Opens the collection ``settings.chroma_collection`` in ``settings.chroma_dir`` with the
    embeddings of ``get_embeddings(settings)``, without creating it, and checks that it was
    built with the same embedding provider and model before the model is loaded. Apart from
    the embedding model, nothing is loaded into memory up front. Loading that model takes
    seconds, so the RAG subgraph calls this function once per compiled graph, on its first
    query (see ``agentic_rag.rag.graph``).

    Args:
        settings: ``chroma_dir`` and ``chroma_collection`` locate the index;
            ``embedding_provider`` and ``embedding_model`` select the query embeddings.

    Returns:
        The Chroma vector store. ``similarity_search_with_relevance_scores`` on it returns
        scores where higher means more relevant.

    Raises:
        IndexNotFoundError: If the collection does not exist yet (run ``agentic-rag ingest``).
            A missing ``chroma_dir`` is not created.
        EmbeddingMismatchError: If the collection was built with another embedding provider
            or model (run ``agentic-rag ingest --rebuild``).
    """
    from langchain_chroma import Chroma

    missing = (
        f"The vector index {settings.chroma_collection!r} does not exist in "
        f"{settings.chroma_dir}; run 'agentic-rag ingest' to build it"
    )
    if not settings.chroma_dir.is_dir():
        raise IndexNotFoundError(missing)
    client = _client(settings.chroma_dir)
    if settings.chroma_collection not in _collection_names(client):
        raise IndexNotFoundError(missing)
    _check_embeddings(client.get_collection(settings.chroma_collection), settings)
    return Chroma(
        client=client,
        collection_name=settings.chroma_collection,
        embedding_function=get_embeddings(settings),
        collection_configuration=COLLECTION_CONFIGURATION,  # type: ignore[arg-type]
        create_collection_if_not_exists=False,
    )


def index_state(settings: Settings) -> IndexState:
    """Tell whether the vector index exists and matches the embedding settings.

    Reads only the collection's metadata and size: the embedding model is not loaded.

    Args:
        settings: ``chroma_dir`` and ``chroma_collection`` locate the index;
            ``embedding_provider`` and ``embedding_model`` are compared with the ones the
            collection was built with.

    Returns:
        ``"missing"`` when the directory or the collection does not exist or the collection
        is empty, ``"mismatch"`` when it was built with another embedding provider or model,
        ``"ready"`` otherwise.
    """
    if not settings.chroma_dir.is_dir():
        return "missing"
    client = _client(settings.chroma_dir)
    if settings.chroma_collection not in _collection_names(client):
        return "missing"
    collection = client.get_collection(settings.chroma_collection)
    if collection.count() == 0:
        return "missing"
    try:
        _check_embeddings(collection, settings)
    except EmbeddingMismatchError:
        return "mismatch"
    return "ready"


def _client(chroma_dir: Path) -> "ClientAPI":
    """Open the persistent Chroma client of ``chroma_dir``, without telemetry."""
    import chromadb

    return chromadb.PersistentClient(
        path=str(chroma_dir), settings=chromadb.Settings(anonymized_telemetry=False)
    )


def _collection_names(client: "ClientAPI") -> set[str]:
    """The names of the client's collections."""
    return {collection.name for collection in client.list_collections()}


def _check_embeddings(collection: "Collection", settings: Settings) -> None:
    """Raise :class:`EmbeddingMismatchError` when the collection was built with other embeddings."""
    metadata: dict[str, Any] = dict(collection.metadata or {})
    provider, model = metadata.get(_PROVIDER_KEY), metadata.get(_MODEL_KEY)
    same_provider = provider == settings.embedding_provider
    same_model = settings.embedding_provider == "fake" or model == settings.embedding_model
    if same_provider and same_model:
        return
    msg = (
        f"The vector index {collection.name!r} in {settings.chroma_dir} was built with "
        f"EMBEDDING_PROVIDER={provider} and EMBEDDING_MODEL={model}, but the settings say "
        f"{settings.embedding_provider} and {settings.embedding_model}. Vectors of different "
        "models are not comparable: run 'agentic-rag ingest --rebuild', or restore the "
        "settings the index was built with."
    )
    raise EmbeddingMismatchError(msg)


def _add_chunks(collection: "Collection", chunks: Sequence[Document], settings: Settings) -> None:
    """Embed the chunks and write them to the collection, batch by batch."""
    embeddings = get_embeddings(settings)
    done = 0
    for batch in _batches(chunks, EMBEDDING_BATCH_SIZE):
        texts = [chunk.page_content for chunk in batch]
        collection.upsert(
            ids=[chunk.metadata["chunk_id"] for chunk in batch],
            embeddings=embeddings.embed_documents(texts),  # type: ignore[arg-type]
            metadatas=[chunk.metadata for chunk in batch],
            documents=texts,
        )
        done += len(batch)
        logger.info("Embedded %d of %d new chunks", done, len(chunks))


def _batches[T](items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    """Consecutive slices of at most ``size`` items."""
    for start in range(0, len(items), size):
        yield items[start : start + size]
