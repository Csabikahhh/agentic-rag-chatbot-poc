"""Tests for the ingestion pipeline: sources, download, Markdown cleaning, loaders, chunking, index.

Everything runs offline: the download tests fetch from a git repository created in a temporary
directory (they are skipped when git is not installed), and the index tests use the fake
hashing embeddings with a Chroma index under ``tmp_path``. The repository's own source list is
validated, but not downloaded.
"""

import json
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pydantic import ValidationError

from agentic_rag.agent.compat import BCD_DIRECTORY
from agentic_rag.config import Settings
from agentic_rag.embeddings import HashingEmbeddings
from agentic_rag.errors import ConfigurationError
from agentic_rag.ingestion import download, index, prepare
from agentic_rag.ingestion.chunking import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CODE_BLOCK_LIMIT,
    DEFAULT_SEPARATORS,
    ChunkingConfig,
    ChunkMetadata,
    context_line,
    split_documents,
)
from agentic_rag.ingestion.download import (
    DownloadError,
    DownloadStats,
    SourceDownload,
    download_sources,
)
from agentic_rag.ingestion.index import (
    EmbeddingMismatchError,
    IndexNotFoundError,
    IndexStats,
    build_index,
    index_state,
    load_index,
)
from agentic_rag.ingestion.loaders import (
    DocumentMetadata,
    list_corpus_files,
    load_documents,
    load_file,
)
from agentic_rag.ingestion.markdown import parse_markdown, split_front_matter
from agentic_rag.ingestion.prepare import prepare_knowledge_base
from agentic_rag.ingestion.sources import (
    DEFAULT_SOURCES_FILE,
    MANIFEST_NAME,
    CorpusSource,
    SourceManifest,
    glob_regex,
    load_sources,
    read_manifest,
    sparse_directories,
)
from agentic_rag.rag.state import Source

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
COMMIT = "0123456789abcdef0123456789abcdef01234567"


def write(path: Path, text: str) -> Path:
    """Write a UTF-8 file, creating its directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def corpus_source(**changes: Any) -> CorpusSource:
    """A valid source, with ``changes`` applied."""
    return CorpusSource.model_validate(
        {
            "id": "react",
            "name": "React",
            "repository": "https://github.com/reactjs/react.dev",
            "commit": COMMIT,
            "license": "CC-BY-4.0",
            "root": "src/content",
            "include": ["learn/**/*.md"],
            "url": "https://react.dev/{path}",
            **changes,
        }
    )


def install_source(data_dir: Path, source: CorpusSource, files: dict[str, str]) -> None:
    """Put a downloaded source in place: its files and its manifest."""
    for relative, text in files.items():
        write(data_dir / source.id / relative, text)
    manifest = SourceManifest.for_source(source, files=len(files))
    write(data_dir / source.id / MANIFEST_NAME, manifest.model_dump_json())


# --- the source list -----------------------------------------------------------------------


def test_the_repository_source_list_is_valid() -> None:
    sources = load_sources(REPOSITORY_ROOT / "data" / "sources.toml")

    assert [source.id for source in sources] == [
        "mdn",
        "react",
        "vue",
        "nextjs",
        "nuxt",
        "typescript",
        "browser-compat-data",
    ]
    for source in sources:
        assert source.repository.startswith("https://github.com/")
        assert sparse_directories(source), "every source checks out only part of its repository"
        if source.index:
            assert source.url is not None and source.url.startswith("https://")
    # The tool data is downloaded like the corpus but never indexed.
    assert [source.id for source in sources if not source.index] == [BCD_DIRECTORY]


@pytest.mark.parametrize(
    ("pattern", "path", "matches"),
    [
        ("**/index.md", "index.md", True),
        ("**/index.md", "web/css/index.md", True),
        ("learn/*.md", "learn/state.md", True),
        ("learn/*.md", "learn/deep/state.md", False),
        ("learn/**/*.md", "learn/state.md", True),
        ("learn/**/*.md", "learn/deep/state.md", True),
        ("*.md", "learn/state.md", False),
        ("a?c.md", "abc.md", True),
        ("a?c.md", "a/c.md", False),
        ("3.guide/3.ai/**", "3.guide/3.ai/mcp.md", True),
        ("reference/Utility Types.md", "reference/Utility Types.md", True),
        ("reference/Utility Types.md", "reference/UtilityXTypes.md", False),
    ],
)
def test_glob_patterns_match_whole_paths(pattern: str, path: str, matches: bool) -> None:
    assert (glob_regex(pattern).fullmatch(path) is not None) is matches


def test_a_source_selects_included_files_that_are_not_excluded() -> None:
    source = corpus_source(include=["guide/**/*.md", "api/*.md"], exclude=["api/index.md"])

    assert source.selects("guide/essentials/computed.md")
    assert source.selects("api/application.md")
    assert not source.selects("api/index.md")
    assert not source.selects("tutorial/step-1.md")
    assert not source.selects("guide/logo.png")


@pytest.mark.parametrize(
    ("changes", "path", "front_matter", "url"),
    [
        ({}, "learn/thinking-in-react.md", {}, "https://react.dev/learn/thinking-in-react"),
        ({}, "learn/index.md", {}, "https://react.dev/learn"),
        (
            {"url": "https://nextjs.org/docs/app/{path}", "strip_order_prefixes": True},
            "01-getting-started/06-fetching-data.mdx",
            {},
            "https://nextjs.org/docs/app/getting-started/fetching-data",
        ),
        (
            {"url": "https://nuxt.com/docs/{path}", "strip_order_prefixes": True},
            "4.api/2.composables/use-state.md",
            {},
            "https://nuxt.com/docs/api/composables/use-state",
        ),
        (
            {"url": "https://vuejs.org/{path}.html"},
            "guide/essentials/computed.md",
            {},
            "https://vuejs.org/guide/essentials/computed.html",
        ),
        (
            {"url": "https://developer.mozilla.org/en-US/docs/{slug}"},
            "web/css/reference/selectors/_colon_has/index.md",
            {"slug": "Web/CSS/Reference/Selectors/:has"},
            "https://developer.mozilla.org/en-US/docs/Web/CSS/Reference/Selectors/:has",
        ),
        ({"url": "https://developer.mozilla.org/{slug}"}, "a/index.md", {}, None),
        ({"url": None}, "learn/state.md", {}, None),
    ],
)
def test_page_urls_follow_the_template(
    changes: dict[str, Any], path: str, front_matter: dict[str, str], url: str | None
) -> None:
    assert corpus_source(**changes).page_url(path, front_matter) == url


def test_order_prefixes_stay_unless_the_source_strips_them() -> None:
    source = corpus_source(url="https://example.org/{path}")

    assert source.page_url("01-intro/02-setup.md", {}) == "https://example.org/01-intro/02-setup"


def test_sparse_directories_are_the_literal_prefixes_below_the_root() -> None:
    source = corpus_source(
        root="files/en-us",
        include=["web/css/**/index.md", "web/css/guides/*.md", "web/html/*/index.md"],
    )

    # web/css/guides is inside web/css; web/html stops at the first wildcard.
    assert sparse_directories(source) == ["files/en-us/web/css", "files/en-us/web/html"]
    assert sparse_directories(corpus_source(root="docs", include=["*.mdx"])) == ["docs"]
    assert sparse_directories(corpus_source(root="", include=["*.md"])) == []


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("[[sources]\nid = 1", "is not valid TOML"),
        ("sources = []", "sources"),
        ('[[sources]]\nid = "react"', "commit"),
    ],
)
def test_an_invalid_source_list_is_a_configuration_error(
    tmp_path: Path, text: str, problem: str
) -> None:
    path = write(tmp_path / "sources.toml", text)

    with pytest.raises(ConfigurationError) as excinfo:
        load_sources(path)

    assert str(path) in str(excinfo.value)
    assert problem in str(excinfo.value)


def test_source_ids_must_be_unique(tmp_path: Path) -> None:
    entry = corpus_source().model_dump(exclude_none=True)
    table = "\n".join(f"{key} = {json.dumps(value)}" for key, value in entry.items())
    path = write(tmp_path / "sources.toml", f"[[sources]]\n{table}\n\n[[sources]]\n{table}\n")

    with pytest.raises(ConfigurationError, match="duplicate source ids: react"):
        load_sources(path)


@pytest.mark.parametrize(
    "changes",
    [
        {"id": "React"},
        {"commit": "main"},
        {"root": "../outside"},
        {"include": []},
        {"include": ["/absolute/*.md"]},
        {"exclude": ["../*.md"]},
        {"unknown": "key"},
    ],
)
def test_invalid_sources_are_rejected(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="validation error"):
        corpus_source(**changes)


def test_a_manifest_records_its_source(tmp_path: Path) -> None:
    source = corpus_source()
    install_source(tmp_path, source, {"learn/state.md": "# State\n\nText."})

    manifest = read_manifest(tmp_path / "react")

    assert manifest is not None
    assert manifest.files == 1
    assert manifest.source() == source
    assert read_manifest(tmp_path / "missing") is None


def test_an_invalid_manifest_is_an_error(tmp_path: Path) -> None:
    write(tmp_path / "react" / MANIFEST_NAME, "{not json")

    with pytest.raises(ValueError, match="not a valid source manifest"):
        read_manifest(tmp_path / "react")


# --- the download --------------------------------------------------------------------------

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def git(*args: str, cwd: Path) -> str:
    """Run git in ``cwd`` with a fixed identity; return its output."""
    command = ["git", "-c", "user.name=Test", "-c", "user.email=test@example.org", *args]
    result = subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


class Upstream:
    """A local git repository that plays the documentation repository."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.mkdir()
        git("init", "--quiet", cwd=path)
        # A local server must allow fetching a commit by its id with a filter, like GitHub.
        git("config", "uploadpack.allowAnySHA1InWant", "true", cwd=path)
        git("config", "uploadpack.allowFilter", "true", cwd=path)

    @property
    def url(self) -> str:
        """The repository URL for a source."""
        return self.path.as_uri()

    def commit(self, files: dict[str, str | None]) -> str:
        """Write (or, for None, delete) files and commit them; return the commit id."""
        for relative, text in files.items():
            if text is None:
                (self.path / relative).unlink()
            else:
                write(self.path / relative, text)
        git("add", "--all", cwd=self.path)
        git("commit", "--quiet", "-m", "update", cwd=self.path)
        return git("rev-parse", "HEAD", cwd=self.path)


@pytest.fixture
def upstream(tmp_path: Path) -> Iterator[Upstream]:
    """An empty upstream repository."""
    yield Upstream(tmp_path / "upstream")


def write_sources(path: Path, *sources: CorpusSource) -> Path:
    """Write a source list in the TOML format of ``data/sources.toml``."""
    tables = []
    for source in sources:
        entry = source.model_dump(exclude_none=True)
        tables.append(
            "[[sources]]\n"
            + "\n".join(f"{key} = {json.dumps(value)}" for key, value in entry.items())
        )
    return write(path, "\n\n".join(tables) + "\n")


@needs_git
def test_download_copies_the_selected_files_of_the_pinned_commit(
    settings: Settings, upstream: Upstream, tmp_path: Path
) -> None:
    commit = upstream.commit(
        {
            "README.md": "# Repository\n",
            "docs/guide/state.md": "# State\n\nText.\n",
            "docs/guide/deep/effects.md": "# Effects\n\nText.\n",
            "docs/guide/logo.png": "not markdown",
            "docs/guide/draft.md": "# Draft\n",
            "docs/blog/news.md": "# News\n",
        }
    )
    source = corpus_source(
        id="guide",
        repository=upstream.url,
        commit=commit,
        root="docs",
        include=["guide/**/*.md"],
        exclude=["guide/draft.md"],
    )
    sources_file = write_sources(tmp_path / "sources.toml", source)

    stats = download_sources(settings, sources_file=sources_file)

    target = settings.data_dir / "guide"
    copied = sorted(path.relative_to(target).as_posix() for path in target.rglob("*"))
    assert copied == [
        MANIFEST_NAME,
        "guide",
        "guide/deep",
        "guide/deep/effects.md",
        "guide/state.md",
    ]
    manifest = read_manifest(target)
    assert manifest is not None and manifest.source() == source and manifest.files == 2
    assert [(s.id, s.files, s.skipped) for s in stats.sources] == [("guide", 2, False)]
    # Nothing but the source directory is left behind in DATA_DIR.
    assert [path.name for path in settings.data_dir.iterdir()] == ["guide"]


@needs_git
def test_download_skips_an_up_to_date_source_and_replaces_an_outdated_one(
    settings: Settings, upstream: Upstream, tmp_path: Path
) -> None:
    first = upstream.commit({"docs/a.md": "# A\n", "docs/b.md": "# B\n"})
    source = corpus_source(
        id="docs", repository=upstream.url, commit=first, root="docs", include=["*.md"]
    )
    download_sources(settings, sources_file=write_sources(tmp_path / "one.toml", source))

    again = download_sources(settings, sources_file=write_sources(tmp_path / "one.toml", source))
    second = upstream.commit({"docs/b.md": None, "docs/c.md": "# C\n"})
    newer = source.model_copy(update={"commit": second})
    updated = download_sources(settings, sources_file=write_sources(tmp_path / "two.toml", newer))

    assert [s.skipped for s in again.sources] == [True]
    assert [s.skipped for s in updated.sources] == [False]
    names = sorted(path.name for path in (settings.data_dir / "docs").glob("*.md"))
    assert names == ["a.md", "c.md"]


@needs_git
def test_download_fails_when_no_file_matches_and_keeps_the_old_directory(
    settings: Settings, upstream: Upstream, tmp_path: Path
) -> None:
    commit = upstream.commit({"docs/a.md": "# A\n"})
    good = corpus_source(
        id="docs", repository=upstream.url, commit=commit, root="", include=["docs/*.md"]
    )
    download_sources(settings, sources_file=write_sources(tmp_path / "good.toml", good))
    empty = good.model_copy(update={"include": ("docs/*.txt",)})

    with pytest.raises(DownloadError, match="no file"):
        download_sources(settings, sources_file=write_sources(tmp_path / "empty.toml", empty))

    assert (settings.data_dir / "docs" / "docs" / "a.md").is_file()
    assert sorted(path.name for path in settings.data_dir.iterdir()) == ["docs"]


@needs_git
def test_download_reports_a_commit_the_repository_does_not_have(
    settings: Settings, upstream: Upstream, tmp_path: Path
) -> None:
    upstream.commit({"docs/a.md": "# A\n"})
    source = corpus_source(id="docs", repository=upstream.url, commit=COMMIT, include=["*.md"])

    with pytest.raises(DownloadError, match="git fetch"):
        download_sources(settings, sources_file=write_sources(tmp_path / "sources.toml", source))


def test_download_needs_git(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(download.shutil, "which", lambda name: None)
    sources_file = write_sources(tmp_path / "sources.toml", corpus_source())

    with pytest.raises(DownloadError, match="git is required"):
        download_sources(settings, sources_file=sources_file)


# --- Markdown cleaning ---------------------------------------------------------------------


def sections(text: str) -> dict[tuple[str, ...], str]:
    """The sections of a page by path."""
    return {section.path: section.text for section in parse_markdown(text).sections}


def test_front_matter_gives_its_top_level_scalars() -> None:
    text = (
        "---\n"
        'title: "`:has()` CSS \\"pseudo-class\\""\n'
        "slug: Web/CSS/:has\n"
        "short: 'it''s'\n"
        "related:\n"
        "  links:\n"
        "    - app/guides\n"
        "preamble: >\n"
        "  folded text\n"
        "---\n"
        "Body.\n"
    )

    values, body = split_front_matter(text)

    assert values == {
        "title": '`:has()` CSS "pseudo-class"',
        "slug": "Web/CSS/:has",
        "short": "it's",
    }
    assert body == "Body."


@pytest.mark.parametrize("text", ["No front matter.\n", "---\ntitle: never closed\n"])
def test_text_without_a_closed_front_matter_is_all_body(text: str) -> None:
    assert split_front_matter(text) == ({}, text)


def test_a_page_is_split_at_its_h2_and_h3_headings() -> None:
    page = parse_markdown(
        "# useState {/*use-state*/}\n\n"
        "Intro text.\n\n"
        "## Reference {/*reference*/}\n\n"
        "### `useState(initialState)` {#usestate}\n\n"
        "Call it.\n\n"
        "#### Parameters\n\n"
        "* `initialState`: the value.\n\n"
        "## Usage\n\n"
        "Use it.\n"
    )

    assert page.title == "useState"
    assert [(s.path, s.text) for s in page.sections] == [
        ((), "Intro text."),
        (
            ("Reference", "useState(initialState)"),
            "Call it.\n\n#### Parameters\n\n* `initialState`: the value.",
        ),
        (("Usage",), "Use it."),
    ]


def test_the_front_matter_title_and_description_come_first() -> None:
    page = parse_markdown(
        "---\ntitle: Fetching Data\ndescription: Learn how to fetch data.\n---\n\n"
        "# Another heading\n\n## Section\n\nText.\n"
    )

    assert page.title == "Fetching Data"
    assert page.sections[0].path == ()
    assert page.sections[0].text == "Learn how to fetch data."


def test_mdn_macros_links_callouts_and_tables_become_plain_markdown() -> None:
    result = sections(
        '---\ntitle: "`:has()` CSS pseudo-class"\nslug: Web/CSS/:has\n---\n\n'
        "{{CSSRef}}\n\n"
        "The **`:has()`** [pseudo-class](/en-US/docs/Web/CSS/Pseudo-classes) works like "
        '{{cssxref(":is()")}} and {{domxref("Window.fetch()", "fetch()")}} for '
        '{{HTMLElement("section")}} in {{Glossary("CSS")}}. ![A diagram](diagram.png)\n\n'
        "> [!NOTE]\n> Pseudo-elements are not valid.\n\n"
        "- `selector`\n  - : A relative selector. {{deprecated_inline}}\n\n"
        '<table class="standard-table">\n  <tr><td><kbd>Enter</kbd> key</td></tr>\n</table>\n\n'
        "## Specifications\n\n{{Specifications}}\n\n"
        "## Browser compatibility\n\n{{Compat}}\n"
    )

    assert list(result) == [()]
    text = result[()]
    assert (
        "The **`:has()`** pseudo-class works like `:is()` and `fetch()` for `<section>` in CSS."
        in text
    )
    assert "> **Note:**\n> Pseudo-elements are not valid." in text
    assert "- `selector`\n    A relative selector." in text
    assert "Enter key" in text
    for leftover in ("{{", "](", "<table", "<kbd>", "diagram"):
        assert leftover not in text


def test_mdx_components_are_removed_and_their_content_kept() -> None:
    result = sections(
        "<Intro>\n\n`useState` is a Hook.\n\n</Intro>\n\n<InlineToc />\n\n---\n\n"
        "## Usage\n\n"
        "<Sandpack>\n\n```js src/App.js\nexport default function App() {\n"
        "  return <div />;\n}\n```\n\n</Sandpack>\n\n"
        '<Image\n  alt="A diagram"\n  srcLight="/light.png"\n/>\n\n'
        "<Hint>Look inside the braces.</Hint> An `Array<Type>` stays, and so does Array<Type>.\n"
    )

    assert result[()] == "`useState` is a Hook."
    usage = result[("Usage",)]
    assert "```js src/App.js\nexport default function App() {\n  return <div />;\n}\n```" in usage
    assert "Look inside the braces. An `Array<Type>` stays, and so does Array<Type>." in usage
    for leftover in ("<Sandpack", "<Image", "alt=", "srcLight", "<Hint>", "---"):
        assert leftover not in usage


def test_pages_router_content_is_removed_from_the_nextjs_docs() -> None:
    result = sections(
        "<AppOnly>\n\nUse the `app` directory.\n\n</AppOnly>\n\n"
        "<PagesOnly>\n\nUse the `pages` directory.\n\n</PagesOnly>\n\n"
        "Edit <AppOnly>`app/page.tsx`</AppOnly><PagesOnly>`pages/index.tsx`</PagesOnly> now.\n"
    )

    assert result[()] == "Use the `app` directory.\n\nEdit `app/page.tsx` now."


def test_nuxt_mdc_components_become_labels_or_disappear() -> None:
    result = sections(
        "## Usage\n\n"
        "```ts\nconst count = useState('counter')\n```\n\n"
        ':read-more{to="/docs/getting-started/state-management"}\n\n'
        "::warning\nDo not name your function `useState`.\n::\n\n"
        '::read-more{to="/docs/guide"}\nRead the guide.\n::\n\n'
        "::code-group\n```bash [npm]\nnpm install\n```\n::\n\n"
        ':video-accordion{title="A video" videoId="x"}\n'
        "#default\n"
    )

    assert result[("Usage",)] == (
        "```ts\nconst count = useState('counter')\n```\n\n"
        "**Warning:**\nDo not name your function `useState`.\n\n"
        "Read the guide.\n\n"
        "```bash [npm]\nnpm install\n```"
    )


def test_vitepress_markup_is_removed_and_template_syntax_kept() -> None:
    result = sections(
        "<script setup>\nimport Demo from './Demo.vue'\n</script>\n\n"
        "# Computed Properties {#computed-properties}\n\n"
        '<div class="composition-api">\n\nShow `{{ count }}` in the template.\n\n</div>\n\n'
        "::: tip\nPrefer computed properties.\n:::\n\n"
        "::: details Why?\nBecause they cache.\n:::\n\n"
        "<style scoped>\n.demo { color: red; }\n</style>\n"
    )

    assert result[()] == (
        "Show `{{ count }}` in the template.\n\n"
        "**Tip:**\nPrefer computed properties.\n\n"
        "**Why?**\nBecause they cache."
    )


def test_code_blocks_are_kept_verbatim() -> None:
    result = sections(
        "## Example\n\n"
        "```bash\n# not a heading\n{{Compat}} [a](b) <Intro>\n```\n\n"
        "~~~md\n## not a section\n~~~\n\n"
        "```ts twoslash\n// @errors: 2345\nconst x: number = 'a';\n//    ^?\n```\n\n"
        "```ts\n// @ts-expect-error\n```\n"
    )

    assert list(result) == [("Example",)]
    assert result[("Example",)] == (
        "```bash\n# not a heading\n{{Compat}} [a](b) <Intro>\n```\n\n"
        "~~~md\n## not a section\n~~~\n\n"
        "```ts twoslash\nconst x: number = 'a';\n```\n\n"
        "```ts\n// @ts-expect-error\n```"
    )


def test_html_comments_are_removed() -> None:
    result = sections("Before <!-- inline --> after.\n\n<!--\nA multi-line\ncomment\n-->\nLast.\n")

    assert result[()] == "Before  after.\n\nLast."


# --- the loaders ---------------------------------------------------------------------------


def test_list_corpus_files_skips_dotfiles_dot_directories_and_readmes(tmp_path: Path) -> None:
    for name in (
        "b.md",
        "a/z.md",
        "a/b/c.mdx",
        ".gitkeep",
        "react/.hidden.md",
        ".react.download/x.md",
        "README.md",
        "docs/readme",
        "docs/ReadMe.hu.md",
    ):
        write(tmp_path / name, "# Text\n")

    files = list_corpus_files(tmp_path)

    assert [path.relative_to(tmp_path).as_posix() for path in files] == [
        "a/b/c.mdx",
        "a/z.md",
        "b.md",
    ]


def test_list_corpus_files_needs_the_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="ingest --download"):
        list_corpus_files(tmp_path / "missing")


def test_list_corpus_files_skips_sources_that_are_not_indexed(tmp_path: Path) -> None:
    install_source(tmp_path, corpus_source(include=["**/*.md"]), {"learn/state.md": "# State\n"})
    install_source(
        tmp_path,
        corpus_source(id="data", include=["**/*.json"], url=None, index=False),
        {"css/has.json": "{}"},
    )

    files = list_corpus_files(tmp_path)

    assert [path.relative_to(tmp_path).as_posix() for path in files] == ["react/learn/state.md"]


def test_a_downloaded_page_gets_its_source_name_and_url(tmp_path: Path) -> None:
    source = corpus_source(include=["**/*.md"])
    install_source(
        tmp_path,
        source,
        {
            "reference/useState.md": (
                "---\ntitle: useState\n---\n\nIntro.\n\n## Usage\n\n### Basic\n\nText.\n"
            )
        },
    )

    documents = load_file(tmp_path / "react" / "reference" / "useState.md", data_dir=tmp_path)

    url = "https://react.dev/reference/useState"
    assert [(d.page_content, d.metadata) for d in documents] == [
        (
            "Intro.",
            {
                "source": "react/reference/useState.md",
                "title": "useState – React",
                "url": url,
                "revision": COMMIT,
            },
        ),
        (
            "Text.",
            {
                "source": "react/reference/useState.md",
                "title": "useState – React",
                "section": "Usage > Basic",
                "url": url,
                "revision": COMMIT,
            },
        ),
    ]


def test_a_page_without_a_manifest_has_no_url(tmp_path: Path) -> None:
    write(tmp_path / "notes" / "setup.md", "Install it.\n")
    write(tmp_path / "plain.txt", "﻿Plain text.\n")
    write(tmp_path / "empty.md", "---\ntitle: Empty\n---\n")

    assert [d.metadata for d in load_file(tmp_path / "notes" / "setup.md", data_dir=tmp_path)] == [
        {"source": "notes/setup.md", "title": "setup"}
    ]
    (text,) = load_file(tmp_path / "plain.txt", data_dir=tmp_path)
    assert (text.page_content, text.metadata) == (
        "Plain text.",
        {"source": "plain.txt", "title": "plain"},
    )
    assert load_file(tmp_path / "empty.md", data_dir=tmp_path) == []


def test_an_unsupported_file_stops_the_ingestion(tmp_path: Path) -> None:
    path = write(tmp_path / "guide.pdf", "%PDF")

    with pytest.raises(ValueError, match=r"Unsupported corpus file guide\.pdf"):
        load_file(path, data_dir=tmp_path)


def test_load_documents_loads_every_file_in_order(settings: Settings) -> None:
    write(settings.data_dir / "b.md", "## One\n\nText.\n\n## Two\n\nText.\n")
    write(settings.data_dir / "a.md", "Text.\n")

    documents = load_documents(settings)

    assert [(d.metadata["source"], d.metadata.get("section")) for d in documents] == [
        ("a.md", None),
        ("b.md", "One"),
        ("b.md", "Two"),
    ]


# --- chunking ------------------------------------------------------------------------------

SMALL = ChunkingConfig(chunk_size=100, chunk_overlap=20, code_block_limit=200)


def document(text: str, **metadata: Any) -> Document:
    """A loaded section with a title."""
    return Document(
        page_content=text, metadata={"source": "guide.md", "title": "Guide", **metadata}
    )


def slices(chunks: list[Document]) -> list[str]:
    """The chunk texts without their context line."""
    return [chunk.page_content.split("\n\n", 1)[1] for chunk in chunks]


def test_chunks_start_with_the_context_line_and_are_verbatim_slices() -> None:
    text = "\n\n".join(f"Paragraph {number} " + "word " * 12 for number in range(5))
    source = document(text, section="Usage > Basic")

    chunks = split_documents([source], config=SMALL)

    assert len(chunks) > 1
    for chunk in chunks:
        header, piece = chunk.page_content.split("\n\n", 1)
        assert header == "Guide > Usage > Basic" == context_line(chunk.metadata)
        start = chunk.metadata["start_index"]
        assert text[start : start + len(piece)] == piece
        assert len(piece) <= SMALL.chunk_size
        assert chunk.metadata["section"] == "Usage > Basic"


def test_a_code_block_stays_whole_up_to_the_limit() -> None:
    code = "```js\n" + "\n".join(f"const value{n} = {n};" for n in range(8)) + "\n```"
    long_code = "```js\n" + "\n".join(f"const value{n} = {n};" for n in range(20)) + "\n```"
    assert SMALL.chunk_size < len(code) <= SMALL.code_block_limit < len(long_code)

    kept = slices(split_documents([document(f"Intro.\n\n{code}\n\nAfter.")], config=SMALL))
    split = slices(split_documents([document(long_code)], config=SMALL))

    assert code in kept
    assert len(split) > 1
    assert all(len(piece) <= SMALL.chunk_size for piece in split)


def test_a_heading_opens_the_chunk_of_the_text_it_introduces() -> None:
    text = "First paragraph " + "word " * 10 + "\n\n#### Parameters\n\n" + "Details " * 10

    pieces = slices(split_documents([document(text)], config=SMALL))

    assert len(pieces) == 2
    assert pieces[1].startswith("#### Parameters\n\nDetails")
    assert "Parameters" not in pieces[0]


def test_a_short_last_paragraph_is_repeated_as_overlap() -> None:
    text = "A " * 30 + "\n\nShort.\n\n" + "B " * 40

    pieces = slices(split_documents([document(text)], config=SMALL))

    assert len(pieces) == 2
    assert pieces[0].endswith("Short.")
    assert pieces[1].startswith("Short.")


def test_chunk_ids_are_stable_unique_and_follow_the_text() -> None:
    chunks = split_documents([document("Same text."), document("Same text.")])
    again = split_documents([document("Same text."), document("Same text.")])
    edited = split_documents([document("Other text.")])

    ids = [chunk.metadata["chunk_id"] for chunk in chunks]
    assert ids == [chunk.metadata["chunk_id"] for chunk in again]
    assert ids[1] == f"{ids[0]}-2"
    assert edited[0].metadata["chunk_id"] not in ids


def test_documents_without_text_give_no_chunks() -> None:
    assert split_documents([document("   \n\n  ")]) == []


# --- the index -----------------------------------------------------------------------------


class CountingEmbeddings(HashingEmbeddings):
    """The hashing embeddings, counting the passages they embed."""

    embedded: ClassVar[list[int]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed and count."""
        type(self).embedded.append(len(texts))
        return super().embed_documents(texts)


@pytest.fixture
def counting_embeddings(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Let the index use :class:`CountingEmbeddings`; return the batch sizes it embedded."""
    CountingEmbeddings.embedded = []
    monkeypatch.setattr(index, "get_embeddings", lambda settings: CountingEmbeddings())
    return CountingEmbeddings.embedded


def frontend_corpus(data_dir: Path) -> None:
    """Three pages of three downloaded sources, about clearly different topics."""
    install_source(
        data_dir,
        corpus_source(
            id="mdn",
            name="MDN",
            root="files/en-us",
            include=["**/index.md"],
            url="https://developer.mozilla.org/en-US/docs/{slug}",
        ),
        {
            "web/css/reference/selectors/_colon_has/index.md": (
                '---\ntitle: "`:has()` CSS pseudo-class"\n'
                "slug: Web/CSS/Reference/Selectors/:has\n---\n\n"
                "The `:has()` pseudo-class selects a parent element that contains a matching "
                "child.\n\n"
                "## Syntax\n\n```css\nsection:has(.featured) {\n  border: 2px solid blue;\n}\n```\n"
            )
        },
    )
    install_source(
        data_dir,
        corpus_source(include=["**/*.md"]),
        {
            "reference/react/useState.md": (
                "---\ntitle: useState\n---\n\n"
                "`useState` is a React Hook that lets you add a state variable to your component.\n"
            )
        },
    )
    install_source(
        data_dir,
        corpus_source(
            id="nuxt",
            name="Nuxt",
            include=["**/*.md"],
            url="https://nuxt.com/docs/{path}",
            strip_order_prefixes=True,
        ),
        {
            "4.api/2.composables/use-state.md": (
                "---\ntitle: useState\n"
                "description: Shared state that survives server rendering.\n---\n\n"
                "## Usage\n\nThe composable serializes the payload to JSON.\n"
            )
        },
    )


def stored_ids(settings: Settings) -> set[str]:
    """The chunk ids in the settings' collection."""
    store = load_index(settings)
    return set(store.get(include=[])["ids"])  # type: ignore[attr-defined]


def test_build_index_stores_every_chunk_and_a_query_finds_the_right_one(
    settings: Settings, counting_embeddings: list[int]
) -> None:
    frontend_corpus(settings.data_dir)

    stats = build_index(settings)
    store = load_index(settings)
    results = store.similarity_search_with_relevance_scores(
        "Which pseudo-class selects a parent element that contains a child?", k=2
    )

    assert (stats.files, stats.documents, stats.chunks, stats.rebuilt) == (3, 5, 5, False)
    assert stats.embedding_provider == "fake"
    assert sum(counting_embeddings) == 5
    best, score = results[0]
    assert 0 < score <= 1
    assert best.metadata == {
        "source": "mdn/web/css/reference/selectors/_colon_has/index.md",
        "title": ":has() CSS pseudo-class – MDN",
        "url": "https://developer.mozilla.org/en-US/docs/Web/CSS/Reference/Selectors/:has",
        "chunk_id": best.metadata["chunk_id"],
        "revision": COMMIT,
        "start_index": 0,
    }
    assert best.page_content.startswith(
        ":has() CSS pseudo-class – MDN\n\nThe `:has()` pseudo-class"
    )


def test_rebuilding_an_unchanged_corpus_embeds_nothing(
    settings: Settings, counting_embeddings: list[int]
) -> None:
    frontend_corpus(settings.data_dir)
    first = build_index(settings)
    ids = stored_ids(settings)
    counting_embeddings.clear()

    second = build_index(settings)

    assert counting_embeddings == []
    assert second.chunks == first.chunks
    assert stored_ids(settings) == ids


def test_build_index_follows_edits_and_removals(
    settings: Settings, counting_embeddings: list[int]
) -> None:
    frontend_corpus(settings.data_dir)
    build_index(settings)
    before = stored_ids(settings)
    page = settings.data_dir / "react" / "reference" / "react" / "useState.md"
    write(page, "---\ntitle: useState\n---\n\nAn edited description of the state Hook.\n")
    (settings.data_dir / "nuxt" / "4.api" / "2.composables" / "use-state.md").unlink()
    counting_embeddings.clear()

    stats = build_index(settings)
    after = stored_ids(settings)

    assert sum(counting_embeddings) == 1
    assert stats.chunks == len(after) == 3
    assert len(before - after) == 3  # the old React chunk and both Nuxt chunks
    assert len(after - before) == 1


def test_rebuild_deletes_and_rebuilds_the_collection(
    settings: Settings, counting_embeddings: list[int]
) -> None:
    frontend_corpus(settings.data_dir)
    build_index(settings)
    counting_embeddings.clear()

    stats = build_index(settings, rebuild=True)

    assert stats.rebuilt
    assert sum(counting_embeddings) == stats.chunks == len(stored_ids(settings))


def test_an_empty_corpus_never_empties_the_index(settings: Settings) -> None:
    frontend_corpus(settings.data_dir)
    build_index(settings)
    ids = stored_ids(settings)
    for path in list(settings.data_dir.iterdir()):
        shutil.rmtree(path)

    with pytest.raises(FileNotFoundError, match="has no documents"):
        build_index(settings)

    assert stored_ids(settings) == ids


def other_embeddings(settings: Settings, **changes: Any) -> Settings:
    """The settings with other embedding values."""
    return Settings.model_validate({**settings.model_dump(), **changes})


def test_an_index_of_other_embeddings_is_rejected_before_the_model_loads(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    frontend_corpus(settings.data_dir)
    build_index(settings)

    def no_model(settings: Settings) -> Embeddings:
        raise AssertionError("the embedding model was loaded")

    monkeypatch.setattr(index, "get_embeddings", no_model)
    huggingface = other_embeddings(settings, embedding_provider="huggingface")

    with pytest.raises(EmbeddingMismatchError, match="ingest --rebuild"):
        load_index(huggingface)
    with pytest.raises(EmbeddingMismatchError):
        build_index(huggingface)
    assert issubclass(EmbeddingMismatchError, ConfigurationError)


def test_the_fake_provider_ignores_the_embedding_model(settings: Settings) -> None:
    frontend_corpus(settings.data_dir)
    build_index(settings)

    store = load_index(other_embeddings(settings, embedding_model="another/model"))

    assert store.similarity_search("state Hook", k=1)


def test_load_index_does_not_create_a_missing_index(settings: Settings) -> None:
    with pytest.raises(IndexNotFoundError, match="agentic-rag ingest"):
        load_index(settings)
    assert not settings.chroma_dir.exists()

    frontend_corpus(settings.data_dir)
    build_index(settings)
    with pytest.raises(IndexNotFoundError):
        load_index(other_embeddings(settings, chroma_collection="another"))


def test_index_state_tells_missing_ready_and_mismatched_indexes(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_model(settings: Settings) -> Embeddings:
        raise AssertionError("the embedding model was loaded")

    assert index_state(settings) == "missing"  # No index directory yet.
    frontend_corpus(settings.data_dir)
    build_index(settings)
    index._client(settings.chroma_dir).create_collection("empty")
    monkeypatch.setattr(index, "get_embeddings", no_model)

    assert index_state(settings) == "ready"
    assert index_state(other_embeddings(settings, chroma_collection="another")) == "missing"
    assert index_state(other_embeddings(settings, chroma_collection="empty")) == "missing"
    huggingface = other_embeddings(settings, embedding_provider="huggingface")
    assert index_state(huggingface) == "mismatch"


# --- start-up preparation ------------------------------------------------------------------


def stand_in_download(monkeypatch: pytest.MonkeyPatch, *, changed: bool) -> list[Path]:
    """Replace the corpus download of the preparation; return the source lists it was given.

    The stand-in reports one source, downloaded (``changed``) or skipped as up to date.
    """
    calls: list[Path] = []

    def download_sources(settings: Settings, *, sources_file: Path) -> DownloadStats:
        calls.append(sources_file)
        source = SourceDownload(id="react", commit="a" * 40, files=1, skipped=not changed)
        return DownloadStats(sources=(source,), duration_ms=1.0)

    monkeypatch.setattr(prepare, "download_sources", download_sources)
    return calls


def test_prepare_knowledge_base_downloads_the_corpus_then_builds_a_missing_index(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    calls = stand_in_download(monkeypatch, changed=True)
    frontend_corpus(settings.data_dir)  # What the download would have put there.
    caplog.set_level("INFO", logger=prepare.__name__)

    stats = prepare_knowledge_base(settings)

    assert calls == [DEFAULT_SOURCES_FILE]
    assert (stats.chunks > 0, stats.rebuilt) == (True, False)
    assert index_state(settings) == "ready"
    messages = [record.getMessage() for record in caplog.records]
    assert "Downloaded the corpus sources react" in messages
    assert any(message.startswith("Building the vector index") for message in messages)


def test_prepare_knowledge_base_embeds_nothing_for_an_unchanged_corpus(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, counting_embeddings: list[int]
) -> None:
    stand_in_download(monkeypatch, changed=False)
    frontend_corpus(settings.data_dir)
    first = build_index(settings)
    embedded = list(counting_embeddings)

    stats = prepare_knowledge_base(settings)

    assert counting_embeddings == embedded  # A later start embeds nothing.
    assert (stats.chunks, stats.rebuilt) == (first.chunks, False)


def test_prepare_knowledge_base_rebuilds_an_index_of_other_embeddings(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    stand_in_download(monkeypatch, changed=False)
    builds: list[bool] = []

    def build_index(settings: Settings, *, rebuild: bool) -> str:
        builds.append(rebuild)
        return "stats"

    monkeypatch.setattr(prepare, "build_index", build_index)
    monkeypatch.setattr(prepare, "index_state", lambda settings: "mismatch")

    assert prepare_knowledge_base(settings) == "stats"
    assert builds == [True]
    assert any("built with other embeddings" in record.getMessage() for record in caplog.records)


def test_prepare_knowledge_base_lets_a_failed_download_propagate(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def download_sources(settings: Settings, *, sources_file: Path) -> DownloadStats:
        raise DownloadError("git fetch failed")

    monkeypatch.setattr(prepare, "download_sources", download_sources)

    with pytest.raises(DownloadError, match="git fetch failed"):
        prepare_knowledge_base(settings)
    assert not settings.chroma_dir.exists()


# --- data contracts ------------------------------------------------------------------------


def valid_index_stats(**changes: Any) -> dict[str, Any]:
    """Arguments of a valid ``IndexStats``, with ``changes`` applied."""
    return {
        "collection": "documents",
        "chroma_dir": Path("data/chroma_db"),
        "embedding_provider": "huggingface",
        "embedding_model": "intfloat/multilingual-e5-small",
        "files": 3,
        "documents": 12,
        "chunks": 40,
        "rebuilt": True,
        "duration_ms": 1234.5,
        **changes,
    }


def test_index_stats_round_trips_through_json() -> None:
    stats = IndexStats(**valid_index_stats())

    assert IndexStats.model_validate_json(stats.model_dump_json()) == stats
    assert stats.chunking == ChunkingConfig()
    assert json.loads(stats.model_dump_json(indent=2))["chunks"] == 40


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("collection", ""),
        ("embedding_provider", "openai"),
        ("embedding_model", ""),
        ("files", -1),
        ("documents", -1),
        ("chunks", -1),
        ("duration_ms", -0.5),
    ],
)
def test_index_stats_rejects_invalid_values(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        IndexStats(**valid_index_stats(**{field: value}))


def test_index_stats_is_immutable() -> None:
    stats = IndexStats(**valid_index_stats())

    with pytest.raises(ValidationError):
        stats.chunks = 0  # type: ignore[misc]


def test_index_errors_extend_the_builtin_errors() -> None:
    assert issubclass(IndexNotFoundError, FileNotFoundError)
    assert issubclass(EmbeddingMismatchError, ValueError)


def test_default_chunking_is_inside_the_planned_range() -> None:
    # Plan section 5.4: start at about 800-1000 characters with 10-20 % overlap.
    assert 800 <= DEFAULT_CHUNK_SIZE <= 1000
    assert 0.10 <= DEFAULT_CHUNK_OVERLAP / DEFAULT_CHUNK_SIZE <= 0.20
    assert ChunkingConfig() == ChunkingConfig(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        code_block_limit=DEFAULT_CODE_BLOCK_LIMIT,
        separators=DEFAULT_SEPARATORS,
    )
    assert DEFAULT_CODE_BLOCK_LIMIT == 2 * DEFAULT_CHUNK_SIZE
    assert DEFAULT_SEPARATORS[-1] == ""


@pytest.mark.parametrize(
    "changes",
    [
        {"chunk_size": 100, "chunk_overlap": 100},
        {"chunk_size": 100, "chunk_overlap": 150},
        {"chunk_size": 0},
        {"chunk_overlap": -1},
        {"chunk_size": 1000, "code_block_limit": 999},
        {"separators": ()},
    ],
)
def test_chunking_config_rejects_invalid_values(changes: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ChunkingConfig(**changes)


def test_chunking_config_is_immutable() -> None:
    config = ChunkingConfig()

    with pytest.raises(ValidationError):
        config.chunk_size = 10  # type: ignore[misc]


def test_document_metadata_keys_are_source_fields() -> None:
    keys = DocumentMetadata.__required_keys__ | DocumentMetadata.__optional_keys__

    assert keys == {"source", "title", "page", "section", "url", "revision"}
    assert keys <= set(Source.model_fields)
    assert DocumentMetadata.__required_keys__ == {"source"}


def test_chunk_metadata_fills_every_source_field_except_content_and_score() -> None:
    keys = ChunkMetadata.__required_keys__ | ChunkMetadata.__optional_keys__

    assert ChunkMetadata.__required_keys__ == {"source", "chunk_id"}
    assert keys - {"start_index"} == set(Source.model_fields) - {"content", "score"}
