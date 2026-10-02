"""Chunking: split the loaded sections into chunks for embedding (plan section 12.6).

Second step of the ingestion pipeline (plan section 5.4). The loaders already cut every page at
its H2 and H3 headings, so a document is one section. :func:`split_documents` cuts each
section into chunks of about ``chunk_size`` characters without breaking its structure:

1. The section text is divided into blocks: fenced code blocks, heading lines (H4 and
   deeper stay in the section text), and the paragraphs between blank lines (a list or a
   table is one paragraph).
2. Consecutive blocks are packed into a chunk while the chunk stays within ``chunk_size``.
   The next chunk starts again with the trailing paragraphs of the previous one that fit in
   ``chunk_overlap``, so a sentence that leads into the next paragraph is not lost. A heading
   never ends a chunk: it moves to the next one, with the text it introduces.
3. A code block longer than ``chunk_size`` stays whole, up to ``code_block_limit``
   characters: an example cut in half explains nothing. Only longer blocks, and paragraphs
   longer than ``chunk_size``, are split by LangChain's ``RecursiveCharacterTextSplitter``
   (:func:`build_splitter`), at the first of ``separators`` that gives short enough pieces,
   with ``chunk_overlap`` characters repeated across every cut.

Each chunk is a verbatim slice of its section, and its text starts with a context line: the
page title and the section headings, ``useState – React > Reference > useState(initialState)``.
A slice alone often lacks the subject ("Parameters", "Returns"), and the line gives it to the
embedding model and to the answer prompt. ``start_index`` is the offset of the slice in the
section text.

The defaults are a starting point inside the plan's range of 800-1000 characters with 10-20 %
overlap: 900 characters (roughly 200-300 tokens, well below the 512-token input limit of the
E5 models) with 150 characters of overlap (about 17 %). Smaller chunks also keep the answer
prompt short when several sub-tasks each contribute ``TOP_K`` chunks. The evaluation (Phase 7)
measures them.

Every chunk keeps the :class:`~agentic_rag.ingestion.loaders.DocumentMetadata` of its
document and adds :class:`ChunkMetadata`.

``langchain_text_splitters`` imports sentence-transformers and torch in its package
``__init__`` (several seconds), so this module imports it only inside the functions that
split, never at module level.
"""

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Required, Self

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentic_rag.ingestion.loaders import SECTION_SEPARATOR, DocumentMetadata

if TYPE_CHECKING:
    from langchain_text_splitters import TextSplitter

__all__ = [
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CODE_BLOCK_LIMIT",
    "DEFAULT_SEPARATORS",
    "ChunkMetadata",
    "ChunkingConfig",
    "build_splitter",
    "context_line",
    "split_documents",
]

DEFAULT_CHUNK_SIZE: Final = 900
"""Maximum chunk length in characters, without the context line."""

DEFAULT_CHUNK_OVERLAP: Final = 150
"""Characters shared by neighbouring chunks, about 17 % of :data:`DEFAULT_CHUNK_SIZE`."""

DEFAULT_CODE_BLOCK_LIMIT: Final = 1800
"""Longest code block kept whole: twice :data:`DEFAULT_CHUNK_SIZE`.

About 500-600 tokens of code, close to the 512-token input limit of the E5 models; the model
reads the beginning of a longer chunk only.
"""

DEFAULT_SEPARATORS: Final[tuple[str, ...]] = ("\n\n", "\n", " ", "")
"""Cut points in order of preference: paragraphs, lines, words, then anywhere.

These are the defaults of ``RecursiveCharacterTextSplitter``. They only apply inside a single
block that is too long, so lines come before words and code is cut between its lines.
"""

_FENCE: Final = re.compile(r"^\s*(?P<fence>`{3,}|~{3,})")
_HEADING: Final = re.compile(r"^#{1,6}\s")


class ChunkingConfig(BaseModel):
    """Configuration of the chunking.

    Attributes:
        chunk_size: Maximum chunk length in characters, without the context line.
        chunk_overlap: Characters repeated from the end of a chunk at the start of the next
            one; smaller than ``chunk_size``.
        code_block_limit: Longest code block that stays whole in one chunk; at least
            ``chunk_size``.
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
    code_block_limit: int = Field(
        default=DEFAULT_CODE_BLOCK_LIMIT,
        ge=1,
        description="Longest code block kept whole; at least chunk_size.",
    )
    separators: tuple[str, ...] = Field(
        default=DEFAULT_SEPARATORS,
        min_length=1,
        description="Cut points in order of preference.",
    )

    @model_validator(mode="after")
    def _check_sizes(self) -> Self:
        """Reject an overlap that is not smaller than the chunk size, and a code-block limit
        below it.
        """
        if self.chunk_overlap >= self.chunk_size:
            msg = (
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size})"
            )
            raise ValueError(msg)
        if self.code_block_limit < self.chunk_size:
            msg = (
                f"code_block_limit ({self.code_block_limit}) must be at least "
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
            ``Source.chunk_id``. It is a hash of the chunk's metadata (source, title, section,
            URL), its position and its text, so re-ingesting an unchanged file gives the same
            ids. An edit gives new ids to the chunks it changes or shifts; ``build_index``
            deletes their old ids.
        start_index: Character offset of the chunk's slice in the text of its document (the
            section), not counting the context line.
    """

    chunk_id: Required[str]
    start_index: int


@dataclass(frozen=True)
class _Block:
    """A paragraph, a heading line or a fenced code block: ``text[start:end]`` of a section."""

    start: int
    end: int
    code: bool
    heading: bool = False

    @property
    def size(self) -> int:
        """Length in characters."""
        return self.end - self.start


def build_splitter(config: ChunkingConfig | None = None) -> "TextSplitter":
    """Create the splitter for blocks that are too long for one chunk.

    Args:
        config: The chunking configuration; defaults to ``ChunkingConfig()``.

    Returns:
        A ``RecursiveCharacterTextSplitter`` with the configured size, overlap and separators
        that records each piece's ``start_index``.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    config = config or ChunkingConfig()
    return RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=list(config.separators),
        add_start_index=True,
    )


def context_line(metadata: Mapping[str, object]) -> str:
    """Return the line that opens every chunk of a document: its title and section headings.

    Args:
        metadata: The document's metadata.

    Returns:
        ``title > heading > heading``, the parts that are known, joined with
        ``loaders.SECTION_SEPARATOR``; empty when the document has neither.
    """
    parts = [metadata.get("title"), metadata.get("section")]
    return SECTION_SEPARATOR.join(str(part) for part in parts if part)


def split_documents(
    documents: Sequence[Document], *, config: ChunkingConfig | None = None
) -> list[Document]:
    """Split documents into chunks that carry :class:`ChunkMetadata`.

    Applies the rules of the module docstring. Every chunk keeps the metadata of its document
    and adds ``chunk_id`` and ``start_index``; its text is the context line, a blank line and
    the slice. Slices without any non-whitespace text are dropped. Chunk ids are unique: when
    two chunks would get the same id (identical metadata, position and text), the later ones
    get a ``-2``, ``-3``, ... suffix.

    Args:
        documents: The loaded documents, as returned by
            :func:`~agentic_rag.ingestion.loaders.load_documents`.
        config: The chunking configuration; defaults to ``ChunkingConfig()``.

    Returns:
        The chunks, document by document and in text order within each document.
    """
    config = config or ChunkingConfig()
    splitter = build_splitter(config)
    chunks: list[Document] = []
    seen: dict[str, int] = {}
    for document in documents:
        metadata = document.metadata
        header = context_line(metadata)
        for start, piece in _pack(document.page_content, config, splitter):
            if not piece.strip():
                continue
            content = f"{header}\n\n{piece}" if header else piece
            chunk_id = _chunk_id(metadata, start, content)
            seen[chunk_id] = seen.get(chunk_id, 0) + 1
            if seen[chunk_id] > 1:
                chunk_id = f"{chunk_id}-{seen[chunk_id]}"
            chunk_metadata: ChunkMetadata = {
                **metadata,
                "chunk_id": chunk_id,
                "start_index": start,
            }
            chunks.append(Document(page_content=content, metadata=dict(chunk_metadata)))
    return chunks


def _pack(text: str, config: ChunkingConfig, splitter: "TextSplitter") -> list[tuple[int, str]]:
    """Cut one section into ``(start_index, slice)`` pairs (steps 2 and 3 of the module)."""
    pieces: list[tuple[int, str]] = []
    current: list[_Block] = []

    def flush() -> None:
        if current:
            pieces.append((current[0].start, text[current[0].start : current[-1].end]))

    for block in _blocks(text):
        if block.size > config.chunk_size and not (
            block.code and block.size <= config.code_block_limit
        ):
            flush()
            current = []
            for piece in splitter.create_documents([text[block.start : block.end]]):
                pieces.append((block.start + piece.metadata["start_index"], piece.page_content))
            continue
        if current and block.end - current[0].start > config.chunk_size:
            # A heading belongs to the text after it: it opens the next chunk instead.
            carried = [current.pop()] if current[-1].heading and len(current) > 1 else []
            flush()
            current = carried or _overlap(current, config.chunk_overlap)
            if current and block.end - current[0].start > config.chunk_size:
                current = []
        current.append(block)
    flush()
    return pieces


def _overlap(blocks: list[_Block], budget: int) -> list[_Block]:
    """The trailing paragraphs of a chunk that fit in ``budget`` characters; no code."""
    kept: list[_Block] = []
    for block in reversed(blocks):
        if block.code or block.heading or blocks[-1].end - block.start > budget:
            break
        kept.insert(0, block)
    return kept


def _blocks(text: str) -> list[_Block]:
    """Divide a section into paragraphs and fenced code blocks (step 1 of the module)."""
    blocks: list[_Block] = []
    start: int | None = None
    code_start = end = 0
    fence: str | None = None
    offset = 0
    for line in text.splitlines(keepends=True):
        line_start, offset = offset, offset + len(line)
        content = line.rstrip("\r\n")
        if fence is not None:
            end = line_start + len(content)
            stripped = content.strip()
            if stripped.startswith(fence) and set(stripped) <= {fence[0]}:
                blocks.append(_Block(code_start, end, code=True))
                fence = None
            continue
        opening = _FENCE.match(content)
        if opening:
            if start is not None:
                blocks.append(_Block(start, end, code=False))
            code_start, end, fence = line_start, line_start + len(content), opening["fence"]
            start = None
            continue
        if not content.strip():
            if start is not None:
                blocks.append(_Block(start, end, code=False))
                start = None
            continue
        if _HEADING.match(content):
            if start is not None:
                blocks.append(_Block(start, end, code=False))
                start = None
            end = line_start + len(content)
            blocks.append(_Block(line_start, end, code=False, heading=True))
            continue
        if start is None:
            start = line_start
        end = line_start + len(content)
    if fence is not None:
        blocks.append(_Block(code_start, end, code=True))
    elif start is not None:
        blocks.append(_Block(start, end, code=False))
    return blocks


def _chunk_id(metadata: Mapping[str, object], start: int, content: str) -> str:
    """Hash a chunk's metadata, position and text into a stable 32-character id."""
    fields = [
        *(str(metadata.get(key, "")) for key in ("source", "title", "section", "url", "page")),
        str(start),
        content,
    ]
    return hashlib.sha256("\x1f".join(fields).encode("utf-8")).hexdigest()[:32]
