"""Chunking: split the loaded documents into overlapping chunks for embedding.

Second step of the ingestion pipeline (plan section 5.4), planned for Phase 2. The splitter is
LangChain's ``RecursiveCharacterTextSplitter``: it cuts at the first separator of
``separators`` that yields pieces of at most ``chunk_size`` characters (paragraphs, then
lines, then words) and repeats up to ``chunk_overlap`` characters of context across every cut.

The defaults are a starting point inside the plan's range of 800-1000 characters with 10-20 %
overlap: 900 characters (roughly 200-300 tokens, well below the 512-token input limit of the
E5 models) with 150 characters of overlap (about 17 %). Phase 2 tunes them against the
evaluation set and records the choice in the README. Smaller chunks also keep the answer
prompt short when several sub-tasks each contribute ``TOP_K`` chunks.

Every chunk keeps the :class:`~agentic_rag.ingestion.loaders.DocumentMetadata` of its
document and adds :class:`ChunkMetadata`.

``langchain_text_splitters`` imports sentence-transformers and torch in its package
``__init__`` (several seconds), so this module imports it only inside the functions that
split, never at module level.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Final, Required, Self

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentic_rag.errors import planned
from agentic_rag.ingestion.loaders import DocumentMetadata

if TYPE_CHECKING:
    from langchain_text_splitters import TextSplitter

__all__ = [
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_SEPARATORS",
    "ChunkMetadata",
    "ChunkingConfig",
    "build_splitter",
    "split_documents",
]

DEFAULT_CHUNK_SIZE: Final = 900
"""Maximum chunk length in characters; a starting point that Phase 2 tunes."""

DEFAULT_CHUNK_OVERLAP: Final = 150
"""Characters shared by neighbouring chunks, about 17 % of :data:`DEFAULT_CHUNK_SIZE`."""

DEFAULT_SEPARATORS: Final[tuple[str, ...]] = ("\n\n", "\n", " ", "")
"""Cut points in order of preference: paragraphs, lines, words, then anywhere.

These are the defaults of ``RecursiveCharacterTextSplitter``; Phase 2 may add sentence
boundaries or format-specific separators.
"""


class ChunkingConfig(BaseModel):
    """Configuration of the text splitter.

    Attributes:
        chunk_size: Maximum chunk length in characters.
        chunk_overlap: Characters repeated from the end of a chunk at the start of the next
            one; smaller than ``chunk_size``.
        separators: Cut points in order of preference. The empty string, as the last entry,
            allows a cut anywhere when no other separator gives a short enough piece.
    """

    model_config = ConfigDict(frozen=True)

    chunk_size: int = Field(
        default=DEFAULT_CHUNK_SIZE, ge=1, description="Maximum chunk length in characters."
    )
    chunk_overlap: int = Field(
        default=DEFAULT_CHUNK_OVERLAP,
        ge=0,
        description="Characters shared by neighbouring chunks; smaller than chunk_size.",
    )
    separators: tuple[str, ...] = Field(
        default=DEFAULT_SEPARATORS,
        min_length=1,
        description="Cut points in order of preference.",
    )

    @model_validator(mode="after")
    def _check_overlap(self) -> Self:
        """Reject an overlap that is not smaller than the chunk size."""
        if self.chunk_overlap >= self.chunk_size:
            msg = (
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size})"
            )
            raise ValueError(msg)
        return self


class ChunkMetadata(DocumentMetadata, total=False):
    """Metadata of a chunk: the metadata of its document plus the identity of the chunk.

    Together with the document keys, ``chunk_id`` fills every ``Source`` field except
    ``content`` (the chunk text) and ``score`` (set at retrieval time).

    Attributes:
        chunk_id: Stable id of the chunk: its id in the Chroma collection and
            ``Source.chunk_id``. It is derived from the chunk's source, position and text, so
            re-ingesting an unchanged file gives the same ids. An edit gives new ids to the
            chunks it changes or shifts; ``build_index`` deletes their old ids.
        start_index: Character offset of the chunk in the text of its document.
    """

    chunk_id: Required[str]
    start_index: int


def build_splitter(config: ChunkingConfig | None = None) -> "TextSplitter":
    """Create the text splitter for a configuration.

    Args:
        config: The splitter configuration; defaults to ``ChunkingConfig()``.

    Returns:
        A ``RecursiveCharacterTextSplitter`` with the configured size, overlap and separators
        that records each chunk's ``start_index``.

    Raises:
        PlannedFeatureError: Until Phase 2 implements the function.
    """
    raise planned(f"{__name__}.build_splitter", 2)


def split_documents(
    documents: Sequence[Document], *, config: ChunkingConfig | None = None
) -> list[Document]:
    """Split documents into chunks that carry :class:`ChunkMetadata`.

    Every chunk keeps the metadata of its document and adds ``chunk_id`` and ``start_index``.
    Chunks without any non-whitespace text are dropped.

    Args:
        documents: The loaded documents, as returned by
            :func:`~agentic_rag.ingestion.loaders.load_documents`.
        config: The splitter configuration; defaults to ``ChunkingConfig()``.

    Returns:
        The chunks, document by document and in text order within each document.

    Raises:
        PlannedFeatureError: Until Phase 2 implements the function.
    """
    raise planned(f"{__name__}.split_documents", 2)
