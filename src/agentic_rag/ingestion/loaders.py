"""Document loaders: the corpus files in ``DATA_DIR`` become LangChain ``Document`` objects.

First step of the ingestion pipeline (plan section 5.4), planned for Phase 2. The corpus is
official frontend documentation (plan decision 8), so the file formats are Markdown and MDX
(plan section 12.6).

Which files belong to the corpus (:func:`list_corpus_files`):

- every regular file under ``settings.data_dir``, recursively, in a stable order (sorted by
  the path relative to ``data_dir``), so that two runs build the same index;
- except dotfiles and everything inside dot-directories (``.gitkeep``, ``.DS_Store``,
  ``.ipynb_checkpoints/``) and README files (``README``, ``README.md``, ``README.hu.md``, in
  any letter case). README files describe the corpus, for example its sources and licenses,
  and are not part of it.

Every loaded ``Document`` carries the extracted text as ``page_content`` and
:class:`DocumentMetadata`. The chunks inherit the metadata, and the UI and the answers cite it
through ``agentic_rag.rag.state.Source``.

The loaders only read: in the container ``data_dir`` is mounted read-only.
"""

from pathlib import Path
from typing import Required, TypedDict

from langchain_core.documents import Document

from agentic_rag.config import Settings
from agentic_rag.errors import planned

__all__ = ["DocumentMetadata", "list_corpus_files", "load_documents", "load_file"]


class DocumentMetadata(TypedDict, total=False):
    """Metadata of a loaded document; each key fills the ``Source`` field of the same name.

    Only ``source`` is required. Unknown values are left out rather than set to None, because
    Chroma never stores None: ``add`` rejects it with a TypeError and ``upsert`` silently drops
    the key. Chroma metadata values are str, int, float, bool or lists of them.

    Attributes:
        source: Path of the file relative to ``data_dir`` with forward slashes, for example
            ``guides/setup.pdf``; the same on every machine and in the container.
        title: Title of the document, for example the PDF title, the first Markdown heading
            or the file name.
        page: 1-based page number in paged formats (PDF): PyPDF's 0-based page index plus 1.
        section: Heading of the section the text belongs to, in formats with headings.
    """

    source: Required[str]
    title: str
    page: int
    section: str


def list_corpus_files(data_dir: Path) -> list[Path]:
    """List the files of the corpus, in a stable order.

    Applies the selection rules of the module docstring: all files under ``data_dir``,
    recursively, except dotfiles, files inside dot-directories and README files.

    Args:
        data_dir: The corpus directory, usually ``settings.data_dir``.

    Returns:
        The files to load, sorted by their path relative to ``data_dir``.

    Raises:
        FileNotFoundError: If ``data_dir`` does not exist or is not a directory.
        PlannedFeatureError: Until Phase 2 implements the function.
    """
    raise planned(f"{__name__}.list_corpus_files", 2)


def load_file(path: Path, *, data_dir: Path) -> list[Document]:
    """Load one corpus file into documents with :class:`DocumentMetadata`.

    Paged formats (PDF) give one document per page with ``page`` set; other formats give one
    document per file, or one per section with ``section`` set where the format has
    headings. The text is cleaned for retrieval (for example whitespace and hyphenation
    artefacts of PDF extraction); Phase 2 decides the details per format.

    Args:
        path: A file returned by :func:`list_corpus_files`.
        data_dir: The corpus directory; ``source`` is ``path`` relative to it.

    Returns:
        The documents of the file in reading order; empty when the file has no text.

    Raises:
        ValueError: If the file format is not supported. The corpus is curated, so an
            unexpected file stops the ingestion instead of being skipped silently.
        PlannedFeatureError: Until Phase 2 implements the function.
    """
    raise planned(f"{__name__}.load_file", 2)


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
        PlannedFeatureError: Until Phase 2 implements the function.
    """
    raise planned(f"{__name__}.load_documents", 2)
