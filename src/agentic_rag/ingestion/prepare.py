"""Start-up preparation: make the corpus and the vector index ready before the UI starts.

``agentic-rag serve``, the command of the container, calls :func:`prepare_knowledge_base`
when ``INGEST_ON_START`` is on. It brings a fresh clone, or an empty volume, to a working
knowledge base without a manual step, and keeps an existing one in step with the image:

1. Download the corpus sources that ``DATA_DIR`` does not hold at their pinned commits
   (``download.download_sources``). Up-to-date sources are skipped without network access.
2. Bring the index up to date with ``index.build_index``. A first build embeds every chunk,
   which takes minutes on the CPU with the Hugging Face model; later starts embed nothing
   when the corpus and the chunking are unchanged. An index built with other embeddings
   (another ``EMBEDDING_PROVIDER`` or ``EMBEDDING_MODEL``) is rebuilt instead of failing,
   because it cannot answer queries with the current settings.
"""

import logging
from pathlib import Path

from agentic_rag.config import Settings
from agentic_rag.ingestion.download import download_sources
from agentic_rag.ingestion.index import IndexStats, build_index, index_state
from agentic_rag.ingestion.sources import DEFAULT_SOURCES_FILE

__all__ = ["prepare_knowledge_base"]

logger = logging.getLogger(__name__)


def prepare_knowledge_base(
    settings: Settings, *, sources_file: Path = DEFAULT_SOURCES_FILE
) -> IndexStats:
    """Download the missing corpus sources, then build or update the vector index.

    Args:
        settings: ``data_dir`` receives the corpus; ``chroma_dir``, ``chroma_collection`` and
            the embedding settings select the index.
        sources_file: The source list, ``data/sources.toml`` by default.

    Returns:
        What the index build loaded and stored; ``rebuilt`` is True when an index built with
        other embeddings was replaced.

    Raises:
        DownloadError: If a source could not be downloaded (no git, no network access).
        FileNotFoundError: If ``sources_file`` does not exist, or the corpus has no documents.
        ConfigurationError: If the source list is invalid.
    """
    downloads = download_sources(settings, sources_file=sources_file)
    changed = [source.id for source in downloads.sources if not source.skipped]
    if changed:
        logger.info("Downloaded the corpus sources %s", ", ".join(changed))

    state = index_state(settings)
    if state == "missing":
        logger.info(
            "Building the vector index %r in %s with EMBEDDING_PROVIDER=%s; the first build "
            "embeds every chunk (minutes on the CPU with the huggingface provider)",
            settings.chroma_collection,
            settings.chroma_dir,
            settings.embedding_provider,
        )
    elif state == "mismatch":
        logger.warning(
            "The vector index %r in %s was built with other embeddings; rebuilding it with "
            "EMBEDDING_PROVIDER=%s and EMBEDDING_MODEL=%s",
            settings.chroma_collection,
            settings.chroma_dir,
            settings.embedding_provider,
            settings.embedding_model,
        )
    else:
        logger.info("Updating the vector index %r to match the corpus", settings.chroma_collection)
    return build_index(settings, rebuild=state == "mismatch")
