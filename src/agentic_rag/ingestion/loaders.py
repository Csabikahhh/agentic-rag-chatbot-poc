"""Document loaders: the corpus files in ``DATA_DIR`` become LangChain ``Document`` objects.

First step of the ingestion pipeline (plan section 5.4). The corpus is official frontend
documentation (plan decision 8), so the formats are Markdown and MDX (plan section 12.6):
``.md``, ``.mdx`` and ``.markdown`` files are parsed by :mod:`agentic_rag.ingestion.markdown`;
``.txt`` files are read as one section of plain text. Any other file stops the ingestion.

Which files belong to the corpus (:func:`list_corpus_files`):

- every regular file under ``settings.data_dir``, recursively, in a stable order (sorted by
  the path relative to ``data_dir``), so that two runs build the same index;
- except dotfiles and everything inside dot-directories (``.gitkeep``, ``.source.json``,
  ``.ipynb_checkpoints/``) and README files (``README``, ``README.md``, ``README.hu.md``, in
  any letter case). README files describe the corpus, for example its sources and licenses,
  and are not part of it;
- and except the source directories whose manifest says ``index = false``: data that a tool
  reads, such as ``browser-compat-data/``.

A file gives one ``Document`` per section with text (see ``markdown.parse_markdown``). Every
document carries the cleaned section text as ``page_content`` and :class:`DocumentMetadata`.
When the file lies in a downloaded source directory (``DATA_DIR/<id>/`` with a
``.source.json`` manifest), the source's name is added to the title (``useState – React``) and
the page's public URL is set. The chunks inherit the metadata, and the UI and the answers cite
it through ``agentic_rag.rag.state.Source``.

The loaders only read: in the container ``data_dir`` is mounted read-only.
"""

import logging
import os
from pathlib import Path
from typing import Final, Required, TypedDict

from langchain_core.documents import Document

from agentic_rag.config import Settings
from agentic_rag.ingestion.markdown import Section, parse_markdown
from agentic_rag.ingestion.sources import SourceManifest, read_manifest

__all__ = [
    "SECTION_SEPARATOR",
    "SUPPORTED_SUFFIXES",
    "DocumentMetadata",
    "list_corpus_files",
    "load_documents",
    "load_file",
]

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES: Final = frozenset({".md", ".mdx", ".markdown", ".txt"})
"""File extensions the loaders read, in any letter case."""

SECTION_SEPARATOR: Final = " > "
"""Separator of the headings in ``DocumentMetadata.section``: ``Usage > Adding state``."""

_README_NAMES: Final = frozenset({"readme", "readme.md", "readme.hu.md"})
_MARKDOWN_SUFFIXES: Final = frozenset({".md", ".mdx", ".markdown"})


class DocumentMetadata(TypedDict, total=False):
    """Metadata of a loaded document; each key fills the ``Source`` field of the same name.

    Only ``source`` is required. Unknown values are left out rather than set to None, because
    Chroma never stores None: ``add`` rejects it with a TypeError and ``upsert`` silently drops
    the key. Chroma metadata values are str, int, float, bool or lists of them.

    Attributes:
        source: Path of the file relative to ``data_dir`` with forward slashes, for example
            ``react/reference/react/useState.md``; the same on every machine and in the
            container. Its first segment is the source id of a downloaded source.
        title: Title of the page (front matter, first H1 or file name), followed by the
            source's name for a downloaded source: ``useState – React``.
        page: 1-based page number in paged formats. No supported format has pages, so the
            loaders never set it; the key stays for corpora with PDF files.
        section: The headings of the section, joined with :data:`SECTION_SEPARATOR`; absent
            for the introduction of a page.
        url: Public URL of the page, for a downloaded source with a URL template.
    """

    source: Required[str]
    title: str
    page: int
    section: str
    url: str


def list_corpus_files(data_dir: Path) -> list[Path]:
    """List the files of the corpus, in a stable order.

    Applies the selection rules of the module docstring: all files under ``data_dir``,
    recursively, except dotfiles, files inside dot-directories, README files and the
    directories of sources that are not indexed.

    Args:
        data_dir: The corpus directory, usually ``settings.data_dir``.

    Returns:
        The files to load, sorted by their path relative to ``data_dir``.

    Raises:
        FileNotFoundError: If ``data_dir`` does not exist or is not a directory.
        ValueError: If a source directory has an invalid manifest.
    """
    if not data_dir.is_dir():
        msg = f"Corpus directory not found: {data_dir} (run 'agentic-rag ingest --download')"
        raise FileNotFoundError(msg)
    files: list[Path] = []
    for directory, subdirectories, names in os.walk(data_dir):
        subdirectories[:] = [name for name in subdirectories if not name.startswith(".")]
        if Path(directory) == data_dir:
            subdirectories[:] = [name for name in subdirectories if _indexed(data_dir / name)]
        for name in names:
            path = Path(directory) / name
            if name.startswith(".") or name.casefold() in _README_NAMES or not path.is_file():
                continue
            files.append(path)
    return sorted(files, key=lambda path: path.relative_to(data_dir).as_posix())


def load_file(path: Path, *, data_dir: Path) -> list[Document]:
    """Load one corpus file into documents with :class:`DocumentMetadata`.

    Markdown and MDX files give one document per section with text (``section`` set for every
    section but the introduction); a text file gives one document. The text is cleaned for
    retrieval by ``markdown.parse_markdown``.

    Args:
        path: A file returned by :func:`list_corpus_files`.
        data_dir: The corpus directory; ``source`` is ``path`` relative to it.

    Returns:
        The documents of the file in reading order; empty when the file has no text.

    Raises:
        ValueError: If the file format is not supported, or the manifest of its source
            directory is invalid. The corpus is curated, so an unexpected file stops the
            ingestion instead of being skipped silently.
    """
    return _load_file(path, data_dir=data_dir, manifests={})


def load_documents(settings: Settings) -> list[Document]:
    """Load the whole corpus from ``settings.data_dir``.

    Calls :func:`load_file` for every file of :func:`list_corpus_files` and logs how many
    files and documents were loaded.

    Args:
        settings: The settings; only ``data_dir`` is used.

    Returns:
        All documents, file by file in the order of :func:`list_corpus_files`.

    Raises:
        FileNotFoundError: If ``settings.data_dir`` does not exist.
        ValueError: If a file has an unsupported format.
    """
    files = list_corpus_files(settings.data_dir)
    manifests: dict[str, SourceManifest | None] = {}
    documents: list[Document] = []
    for path in files:
        documents += _load_file(path, data_dir=settings.data_dir, manifests=manifests)
    logger.info(
        "Loaded %d documents from %d files in %s", len(documents), len(files), settings.data_dir
    )
    return documents


def _load_file(
    path: Path, *, data_dir: Path, manifests: dict[str, SourceManifest | None]
) -> list[Document]:
    """Load one file; ``manifests`` caches the manifest of each source directory."""
    source = path.relative_to(data_dir).as_posix()
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        msg = f"Unsupported corpus file {source}: expected one of {supported}"
        raise ValueError(msg)
    text = path.read_text(encoding="utf-8-sig")

    front_matter: dict[str, str] = {}
    if suffix in _MARKDOWN_SUFFIXES:
        page = parse_markdown(text)
        title = page.title or path.stem
        sections = page.sections
        front_matter = page.front_matter
    else:
        title = path.stem
        sections = (Section(path=(), text=text.strip()),) if text.strip() else ()

    url: str | None = None
    source_id, _, path_in_source = source.partition("/")
    manifest = _manifest(data_dir, source_id, manifests) if path_in_source else None
    if manifest is not None:
        title = f"{title} – {manifest.name}"
        url = manifest.page_url(path_in_source, front_matter)

    documents = []
    for section in sections:
        metadata: DocumentMetadata = {"source": source, "title": title}
        if section.path:
            metadata["section"] = SECTION_SEPARATOR.join(section.path)
        if url is not None:
            metadata["url"] = url
        documents.append(Document(page_content=section.text, metadata=dict(metadata)))
    return documents


def _indexed(directory: Path) -> bool:
    """Whether a top-level directory of the corpus is indexed: no manifest, or ``index``."""
    manifest = read_manifest(directory)
    return manifest is None or manifest.index


def _manifest(
    data_dir: Path, source_id: str, manifests: dict[str, SourceManifest | None]
) -> SourceManifest | None:
    """Return the manifest of ``data_dir/<source_id>/``, read once per directory."""
    if source_id not in manifests:
        manifests[source_id] = read_manifest(data_dir / source_id)
    return manifests[source_id]
