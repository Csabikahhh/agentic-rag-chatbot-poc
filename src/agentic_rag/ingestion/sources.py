"""Corpus sources: the documentation repositories the corpus is downloaded from (plan 12.6).

``data/sources.toml`` lists the sources of the corpus, one ``[[sources]]`` table each, and
:func:`load_sources` reads it into :class:`CorpusSource` records:

- ``id``: the directory of the source under ``DATA_DIR`` (``react`` -> ``data/raw/react/``),
  and the first segment of every ``DocumentMetadata.source`` path it contributes;
- ``name``: the short name shown after each page title, for example ``useState – React``, so
  that pages of the same name from different documentation sets stay apart;
- ``repository`` and ``commit``: the git repository and the pinned commit (40 hex digits), so
  every download gets exactly the same files;
- ``license``: the SPDX identifier of the documentation's license;
- ``root``: the directory inside the repository that the paths are relative to;
- ``include`` and ``exclude``: glob patterns that select the files below ``root``;
- ``url``: the template of a page's public URL (see :meth:`CorpusSource.page_url`);
- ``strip_order_prefixes``: whether the path in the URL drops numeric ordering prefixes such as
  ``01-`` or ``3.``, which some documentation sites keep only in their file names.

Glob patterns are matched against the whole path relative to ``root``, with forward slashes:
``*`` matches within one path segment, ``**/`` any number of directories (also none), and
``?`` one character other than ``/``. A file belongs to the source when it matches an
``include`` pattern and no ``exclude`` pattern.

The download writes the source as JSON to ``<DATA_DIR>/<id>/.source.json``, the manifest
(:class:`SourceManifest`). The corpus directory therefore describes itself: the loaders read
the manifest to give each page its title suffix and its URL, also in the container, where only
``DATA_DIR`` is mounted. The manifest is a dotfile, so it is never indexed.

The module uses only the standard library and pydantic, so the loaders can import it cheaply.
"""

import functools
import json
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Final, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from agentic_rag.errors import ConfigurationError

__all__ = [
    "DEFAULT_SOURCES_FILE",
    "MANIFEST_NAME",
    "CorpusSource",
    "SourceManifest",
    "glob_regex",
    "load_sources",
    "read_manifest",
    "sparse_directories",
]

DEFAULT_SOURCES_FILE: Final = Path("data/sources.toml")
"""The source list, relative to the working directory (the repository root)."""

MANIFEST_NAME: Final = ".source.json"
"""File name of the manifest that the download writes into each source directory."""

_ORDER_PREFIX: Final = re.compile(r"^\d+[.-]")
_PAGE_SUFFIXES: Final = (".md", ".mdx", ".markdown", ".txt")
_WILDCARDS: Final = frozenset("*?")


class CorpusSource(BaseModel):
    """One documentation source of the corpus: a set of files at a pinned commit of a repository.

    Attributes:
        id: Directory of the source under ``DATA_DIR``: lower-case letters, digits and hyphens.
        name: Short name added to every page title, for example ``React``.
        repository: URL (or local path) of the git repository.
        commit: The pinned commit, 40 hexadecimal digits.
        license: SPDX identifier of the license of the documentation text.
        root: Directory inside the repository that the patterns and the copied paths are
            relative to; empty for the repository root.
        include: Glob patterns of the files to copy; at least one.
        exclude: Glob patterns of files to leave out, applied after ``include``.
        url: Template of a page's public URL, or None when the pages have none.
        strip_order_prefixes: Drop numeric ordering prefixes (``01-``, ``3.``) from the path
            segments that the URL template receives.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$", max_length=40)
    name: str = Field(min_length=1)
    repository: str = Field(min_length=1)
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    license: str = Field(min_length=1)
    root: str = ""
    include: tuple[str, ...] = Field(min_length=1)
    exclude: tuple[str, ...] = ()
    url: str | None = None
    strip_order_prefixes: bool = False

    @field_validator("root")
    @classmethod
    def _check_root(cls, value: str) -> str:
        """Normalize the root to a relative POSIX path without leading or trailing slashes."""
        root = value.strip().strip("/")
        if "\\" in root or ".." in PurePosixPath(root).parts:
            msg = "must be a relative path with forward slashes and without '..'"
            raise ValueError(msg)
        return root

    @field_validator("include", "exclude")
    @classmethod
    def _check_patterns(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject empty patterns and patterns that leave the root."""
        for pattern in value:
            if not pattern or pattern.startswith("/") or ".." in pattern.split("/"):
                msg = f"invalid pattern {pattern!r}: use a relative path below 'root'"
                raise ValueError(msg)
        return value

    def selects(self, path: str) -> bool:
        """Return whether a file belongs to the source.

        Args:
            path: The file's path relative to ``root``, with forward slashes.

        Returns:
            True when the path matches an ``include`` pattern and no ``exclude`` pattern.
        """
        return any(glob_regex(p).fullmatch(path) for p in self.include) and not any(
            glob_regex(p).fullmatch(path) for p in self.exclude
        )

    def page_url(self, path: str, front_matter: Mapping[str, str]) -> str | None:
        """Return the public URL of a page, filled in from the ``url`` template.

        The template is a ``str.format`` pattern. ``{path}`` is the page's path relative to
        ``root`` without its file extension and without a final ``index`` segment, with the
        ordering prefixes removed when ``strip_order_prefixes`` is set. Every string value of
        the page's front matter is available by its key too, for example ``{slug}``.

        Args:
            path: The file's path relative to ``root``, with forward slashes.
            front_matter: The scalar values of the page's front matter.

        Returns:
            The URL, or None when the source has no template or the template names a value
            the page does not have.
        """
        if self.url is None:
            return None
        values = {**front_matter, "path": self._url_path(path)}
        try:
            return self.url.format_map(values)
        except (KeyError, IndexError, ValueError):
            return None

    def _url_path(self, path: str) -> str:
        """The ``{path}`` value of the URL template (see :meth:`page_url`)."""
        for suffix in _PAGE_SUFFIXES:
            if path.endswith(suffix):
                path = path.removesuffix(suffix)
                break
        segments = path.split("/")
        if self.strip_order_prefixes:
            segments = [_ORDER_PREFIX.sub("", segment) for segment in segments]
        if segments[-1] == "index":
            segments.pop()
        return "/".join(segments)


class SourceManifest(CorpusSource):
    """What the download wrote into a source directory: the source and its file count.

    Attributes:
        files: Number of files copied from the repository.
    """

    files: int = Field(ge=0)

    def source(self) -> CorpusSource:
        """Return the source this manifest was downloaded from, without the file count."""
        return CorpusSource.model_validate(self.model_dump(exclude={"files"}))

    @classmethod
    def for_source(cls, source: CorpusSource, *, files: int) -> Self:
        """Build the manifest of a downloaded source."""
        return cls.model_validate({**source.model_dump(), "files": files})


class _SourceList(BaseModel):
    """Top level of ``sources.toml``: an array of tables named ``sources``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sources: tuple[CorpusSource, ...] = Field(min_length=1)

    @field_validator("sources")
    @classmethod
    def _check_unique_ids(cls, value: tuple[CorpusSource, ...]) -> tuple[CorpusSource, ...]:
        """Every source needs a directory of its own."""
        ids = [source.id for source in value]
        duplicates = sorted({source_id for source_id in ids if ids.count(source_id) > 1})
        if duplicates:
            msg = f"duplicate source ids: {', '.join(duplicates)}"
            raise ValueError(msg)
        return value


def load_sources(path: Path = DEFAULT_SOURCES_FILE) -> tuple[CorpusSource, ...]:
    """Read and validate the source list.

    Args:
        path: The TOML file, ``data/sources.toml`` by default.

    Returns:
        The sources, in file order.

    Raises:
        FileNotFoundError: If the file does not exist.
        ConfigurationError: If the file is not valid TOML or a source is invalid; the message
            names the file and the problem.
    """
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        return _SourceList.model_validate(data).sources
    except tomllib.TOMLDecodeError as exc:
        msg = f"{path} is not valid TOML: {exc}"
        raise ConfigurationError(msg) from exc
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in issue['loc'])}: {issue['msg']}"
            for issue in exc.errors(include_url=False)
        )
        msg = f"{path} has an invalid source list: {problems}"
        raise ConfigurationError(msg) from exc


def read_manifest(source_dir: Path) -> SourceManifest | None:
    """Read the manifest of a downloaded source directory.

    Args:
        source_dir: A directory under ``DATA_DIR``.

    Returns:
        The manifest, or None when the directory has none (for example a corpus that was put
        in place by hand).

    Raises:
        ValueError: If the manifest exists but is not valid; delete the directory and download
            the source again.
    """
    path = source_dir / MANIFEST_NAME
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        return SourceManifest.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError) as exc:
        msg = f"{path} is not a valid source manifest: {exc}"
        raise ValueError(msg) from exc


@functools.lru_cache(maxsize=512)
def glob_regex(pattern: str) -> re.Pattern[str]:
    """Translate a glob pattern into a regular expression for ``fullmatch``.

    Args:
        pattern: A pattern with ``*``, ``**/`` and ``?`` (see the module docstring).

    Returns:
        The compiled expression.
    """
    parts: list[str] = []
    index = 0
    while index < len(pattern):
        if pattern.startswith("**/", index):
            parts.append("(?:[^/]+/)*")
            index += 3
        elif pattern.startswith("**", index):
            parts.append(".*")
            index += 2
        elif pattern[index] == "*":
            parts.append("[^/]*")
            index += 1
        elif pattern[index] == "?":
            parts.append("[^/]")
            index += 1
        else:
            parts.append(re.escape(pattern[index]))
            index += 1
    return re.compile("".join(parts))


def sparse_directories(source: CorpusSource) -> list[str]:
    """Return the repository directories that hold every file the source can select.

    The download checks out only these directories (git's cone-mode sparse checkout), so a
    large repository is not fetched in full. Each directory is the literal part of an
    ``include`` pattern, before its first wildcard, below ``root``.

    Args:
        source: The source.

    Returns:
        Sorted directories relative to the repository root, without nested duplicates; an
        empty list when a pattern can match anywhere in the repository.
    """
    directories: set[str] = set()
    for pattern in source.include:
        segments = pattern.split("/")
        literal: list[str] = []
        for segment in segments[:-1]:
            if _WILDCARDS & set(segment):
                break
            literal.append(segment)
        directory = "/".join(part for part in (source.root, *literal) if part)
        if not directory:
            return []
        directories.add(directory)
    ordered = sorted(directories)
    return [
        directory
        for directory in ordered
        if not any(directory.startswith(f"{other}/") for other in ordered)
    ]
