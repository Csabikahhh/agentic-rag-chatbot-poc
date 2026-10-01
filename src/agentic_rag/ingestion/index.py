"""Persistent vector index: build and open the Chroma collection (plan section 5.4, decision 6).

Planned for Phase 2. The index is the Chroma collection ``settings.chroma_collection`` in a
persistent client directory, ``settings.chroma_dir`` (``data/chroma_db``: gitignored locally,
a named volume in the container):

- :func:`build_index` runs the pipeline: ``loaders.load_documents`` ->
  ``chunking.split_documents`` -> embed with ``get_embeddings(settings)`` -> upsert into the
  collection under each chunk's ``chunk_id`` -> delete the stored chunks that the run did not
  produce. It backs ``agentic-rag ingest`` and the start-up ingestion of the container
  (``INGEST_ON_START``).
- :func:`load_index` opens the existing collection for the ``retrieve`` node of the RAG
  subgraph.

Both functions take the embedding model from ``agentic_rag.embeddings.get_embeddings``, so the
same settings embed the passages and the queries with the same model. The collection records
``EMBEDDING_PROVIDER`` and ``EMBEDDING_MODEL`` in its metadata, so an index cannot be queried
with another model by mistake. Both providers return unit-length vectors, so Phase 2
configures cosine distance and the relevance scores are ``1 - distance`` (higher is more
relevant, as ``Source.score`` expects).

chromadb and langchain_chroma are imported inside the functions, so importing this module
stays fast.
"""

from pathlib import Path

from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, ConfigDict, Field

from agentic_rag.config import EmbeddingProvider, Settings
from agentic_rag.errors import planned
from agentic_rag.ingestion.chunking import ChunkingConfig

__all__ = [
    "EmbeddingMismatchError",
    "IndexNotFoundError",
    "IndexStats",
    "build_index",
    "load_index",
]


class IndexNotFoundError(FileNotFoundError):
    """The vector index has not been built yet.

    Raised by :func:`load_index`. The message names the directory and the collection and
    tells to run ``agentic-rag ingest``. As a ``FileNotFoundError``, it is also caught by
    callers that only handle missing files.
    """


class EmbeddingMismatchError(ValueError):
    """The vector index was built with another embedding provider or model.

    Vectors of different models are not comparable, so the index has to be rebuilt with
    ``agentic-rag ingest --rebuild``, or the settings have to match the index again.
    """


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

    1. Upsert every chunk the run produced, under its ``chunk_id``.
    2. Once the upsert has succeeded, delete every stored id that the run did not produce:
       the set difference of the stored ids and the produced ids.

    Both steps send batches of at most the client's maximum batch size. The set difference
    handles every change to the corpus: added, edited, shortened, re-chunked and removed
    files. Deletion comes last, so a run that fails without ``rebuild`` never removes chunks.
    An unchanged corpus leaves the collection as it was, so the command is idempotent.

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
        FileNotFoundError: If ``settings.data_dir`` does not exist.
        EmbeddingMismatchError: If the existing collection was built with another embedding
            provider or model and ``rebuild`` is False.
        PlannedFeatureError: Until Phase 2 implements the function.
    """
    raise planned(f"{__name__}.build_index", 2)


def load_index(settings: Settings) -> VectorStore:
    """Open the existing vector index for retrieval.

    Opens the collection ``settings.chroma_collection`` in ``settings.chroma_dir`` with the
    embeddings of ``get_embeddings(settings)``, without creating it, and checks that it was
    built with the same embedding provider and model. Apart from the embedding model, nothing
    is loaded into memory up front. Loading that model takes seconds, so the RAG subgraph
    calls this function once per compiled graph, on its first query (see
    ``agentic_rag.rag.graph``).

    Args:
        settings: ``chroma_dir`` and ``chroma_collection`` locate the index;
            ``embedding_provider`` and ``embedding_model`` select the query embeddings.

    Returns:
        The Chroma vector store. ``similarity_search_with_relevance_scores`` on it returns
        scores where higher means more relevant.

    Raises:
        IndexNotFoundError: If the collection does not exist yet (run ``agentic-rag ingest``).
        EmbeddingMismatchError: If the collection was built with another embedding provider
            or model (run ``agentic-rag ingest --rebuild``).
        PlannedFeatureError: Until Phase 2 implements the function.
    """
    raise planned(f"{__name__}.load_index", 2)
