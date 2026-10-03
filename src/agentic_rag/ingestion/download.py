"""Corpus download: fetch the sources of ``data/sources.toml`` into ``DATA_DIR`` (plan 12.6).

``agentic-rag ingest --download`` calls :func:`download_sources` before building the index.
For every :class:`~agentic_rag.ingestion.sources.CorpusSource` it:

1. Skips the source when ``DATA_DIR/<id>/.source.json`` already records the same source (the
   same commit, root, patterns and URL template), so a repeated run downloads nothing.
2. Checks out the pinned commit with git into a temporary directory: a shallow fetch of that
   one commit with ``--filter=blob:none`` and a cone-mode sparse checkout of the directories
   the ``include`` patterns can match, so only the selected part of a large repository
   (the MDN content repository has 16 000 files) is transferred.
3. Copies the selected files, relative to ``root``, into a staging directory next to the
   target, writes the manifest there and swaps it in place of ``DATA_DIR/<id>/``. Files that
   the new selection no longer contains disappear with the old directory.

The download needs git (2.27 or newer) on the PATH and network access to the repositories. On
the host it runs with ``agentic-rag ingest --download``; the container runs it at start-up
(``agentic_rag.ingestion.prepare``), into the named volume that holds its ``DATA_DIR``.
"""

import logging
import os
import shutil
import stat
import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from agentic_rag.config import Settings
from agentic_rag.ingestion.sources import (
    DEFAULT_SOURCES_FILE,
    MANIFEST_NAME,
    CorpusSource,
    SourceManifest,
    load_sources,
    read_manifest,
    sparse_directories,
)

__all__ = ["DownloadError", "DownloadStats", "SourceDownload", "download_sources"]

logger = logging.getLogger(__name__)

# Byte-identical checkouts on every platform, and paths longer than 260 characters on Windows.
_GIT_OPTIONS = ("-c", "core.autocrlf=false", "-c", "core.longpaths=true")


class DownloadError(RuntimeError):
    """A source could not be downloaded: git is missing, a git command failed, or the
    pinned commit has no file that the ``include`` patterns select.
    """


class SourceDownload(BaseModel):
    """Outcome of one source.

    Attributes:
        id: The source id.
        commit: The commit the directory now holds.
        files: Number of files in the source directory.
        skipped: Whether the directory was already up to date.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    commit: str
    files: int = Field(ge=0)
    skipped: bool


class DownloadStats(BaseModel):
    """Outcome of one :func:`download_sources` run.

    Attributes:
        sources: One entry per source, in the order of the source list.
        duration_ms: Wall-clock duration of the run in milliseconds.
    """

    model_config = ConfigDict(frozen=True)

    sources: tuple[SourceDownload, ...]
    duration_ms: float = Field(ge=0)


def download_sources(
    settings: Settings, *, sources_file: Path = DEFAULT_SOURCES_FILE
) -> DownloadStats:
    """Download every source of the source list into ``settings.data_dir``.

    Args:
        settings: The settings; only ``data_dir`` is used. It is created when missing.
        sources_file: The source list, ``data/sources.toml`` by default.

    Returns:
        What each source contributed and whether it was skipped.

    Raises:
        FileNotFoundError: If ``sources_file`` does not exist.
        ConfigurationError: If the source list is invalid.
        DownloadError: If a source could not be downloaded. The sources before it are
            already in place; a failed source keeps its previous directory.
    """
    started = time.perf_counter()
    sources = load_sources(sources_file)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    results = tuple(_download_source(source, settings.data_dir) for source in sources)
    return DownloadStats(sources=results, duration_ms=(time.perf_counter() - started) * 1000)


def _download_source(source: CorpusSource, data_dir: Path) -> SourceDownload:
    """Bring ``data_dir/<id>/`` to the state the source describes."""
    target = data_dir / source.id
    manifest = read_manifest(target) if target.is_dir() else None
    if manifest is not None and manifest.source() == source:
        logger.info(
            "%s: up to date at %s (%d files)", source.id, source.commit[:12], manifest.files
        )
        return SourceDownload(
            id=source.id, commit=source.commit, files=manifest.files, skipped=True
        )

    logger.info("%s: downloading %s at %s", source.id, source.repository, source.commit[:12])
    workspace = Path(tempfile.mkdtemp(prefix=f"agentic-rag-{source.id}-"))
    staging = data_dir / f".{source.id}.download"
    try:
        checkout = workspace / "repository"
        _checkout(source, checkout)
        _remove_tree(staging)
        files = _copy_selected(source, checkout, staging)
        if files == 0:
            msg = (
                f"{source.id}: no file of {source.repository} at {source.commit} matches the "
                "include patterns"
            )
            raise DownloadError(msg)
        manifest_text = SourceManifest.for_source(source, files=files).model_dump_json(indent=2)
        (staging / MANIFEST_NAME).write_text(manifest_text + "\n", encoding="utf-8")
        _replace_directory(target, staging)
    finally:
        _remove_tree(workspace)
        _remove_tree(staging)
    logger.info("%s: %d files in %s", source.id, files, target)
    return SourceDownload(id=source.id, commit=source.commit, files=files, skipped=False)


def _checkout(source: CorpusSource, checkout: Path) -> None:
    """Check out the selected directories of the pinned commit into ``checkout``."""
    _git("init", "--quiet", str(checkout))
    _git("remote", "add", "origin", source.repository, cwd=checkout)
    directories = sparse_directories(source)
    if directories:
        _git("sparse-checkout", "set", "--cone", "--", *directories, cwd=checkout)
    fetch = ("fetch", "--quiet", "--depth", "1", "--filter=blob:none", "origin", source.commit)
    _git(*fetch, cwd=checkout)
    _git("checkout", "--quiet", "FETCH_HEAD", cwd=checkout)


def _git(*args: str, cwd: Path | None = None) -> None:
    """Run one git command; raise :class:`DownloadError` with git's message when it fails."""
    executable = shutil.which("git")
    if executable is None:
        msg = "git is required to download the corpus; install it and put it on the PATH"
        raise DownloadError(msg)
    command = [executable, *_GIT_OPTIONS, *args]
    try:
        subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        msg = f"git {' '.join(args[:2])} failed with exit code {exc.returncode}: {detail}"
        raise DownloadError(msg) from exc


def _copy_selected(source: CorpusSource, checkout: Path, staging: Path) -> int:
    """Copy the files the source selects, relative to its root; return how many."""
    root = checkout / source.root if source.root else checkout
    copied = 0
    for directory, subdirectories, files in os.walk(root):
        subdirectories[:] = sorted(name for name in subdirectories if name != ".git")
        for name in sorted(files):
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if not source.selects(relative):
                continue
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            copied += 1
    return copied


def _replace_directory(target: Path, staging: Path) -> None:
    """Put ``staging`` in place of ``target``, removing the old directory afterwards."""
    retired = target.with_name(f".{target.name}.old")
    _remove_tree(retired)
    if target.exists():
        target.rename(retired)
    staging.rename(target)
    _remove_tree(retired)


def _remove_tree(path: Path) -> None:
    """Delete a directory tree if it exists, including read-only files (git objects)."""
    if path.exists():
        shutil.rmtree(path, onexc=_make_writable_and_retry)


def _make_writable_and_retry(
    function: Callable[[str], object], path: str, _: BaseException
) -> None:
    """``shutil.rmtree`` error handler: clear the read-only flag that Windows refuses to delete."""
    os.chmod(path, stat.S_IWRITE)
    function(path)
