"""Markdown and MDX pages: front matter, cleaning and sections (plan section 12.6).

The corpus is documentation written in Markdown dialects: MDN's Markdown with KumaScript
macros, MDX with JSX components (React, Next.js), VitePress Markdown with Vue components and
custom containers (Vue), and MDC with block components (Nuxt). :func:`parse_markdown` turns a
page into plain Markdown that reads well in a prompt and embeds well, in three steps:

1. **Front matter.** The YAML block between ``---`` lines at the start of the page is removed;
   its top-level scalar values (``title``, ``slug``, ``permalink``, ``description``, ...) are
   returned as strings. Nested and multi-line values are skipped: no source needs them.
2. **Cleaning**, outside fenced code blocks only; code is kept verbatim, except the twoslash
   directives of TypeScript's examples (``// @errors: 2345``, ``// ^?``):

   - removed with their content: HTML comments, ``<script>`` and ``<style>`` blocks, and the
     ``<PagesOnly>`` blocks of the Next.js documentation (this corpus is the App Router);
   - removed, content kept: lines that hold only a JSX component tag (``<Intro>``,
     ``</Sandpack>``, a multi-line ``<Image ... />``) or a ``<div>`` / ``<span>`` tag, Nuxt's
     MDC block markers (``::tip``, ``::``) and VitePress containers (``::: details``);
   - removed: MDC inline components on their own line (``:read-more{to="..."}``), thematic
     breaks, and MDN macros without text (``{{Compat}}``, ``{{EmbedLiveSample(...)}}``);
   - replaced by their text: links (``[text](url)`` becomes ``text``), MDN cross-reference
     macros (``{{cssxref("color")}}`` becomes ``color``), and outside code spans the HTML
     formatting tags (MDN's ``<table>`` and ``<kbd>``) and inline JSX components
     (``<Hint>text</Hint>``); images are dropped;
   - callouts get a label: MDC's ``::warning`` and GitHub's ``> [!NOTE]`` become
     ``**Warning:**`` and ``> **Note:**``; MDN's definition lists (``- : text``) lose their
     marker; heading anchors (``{/*usage*/}``, ``{#usage}``) are removed.

3. **Sections.** The page is split at its H2 and H3 headings. A section is the text below a
   heading up to the next H2 or H3; its path is the H2 heading, followed by the H3 heading
   inside it. The text before the first H2 is the introduction, with an empty path, and the
   front matter's ``description`` opens it. Deeper headings stay in the text. The first H1 is
   the page title when the front matter has none. Sections without any text are dropped:
   MDN's "Specifications" and "Browser compatibility" sections, for example, hold only a macro.

The functions are pure and work line by line on ``str``; they import nothing heavy.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final

__all__ = ["MarkdownPage", "Section", "parse_markdown", "split_front_matter"]

_FENCE: Final = re.compile(r"^\s*(?P<fence>`{3,}|~{3,})(?P<info>.*)$")
_HEADING: Final = re.compile(r"^(?P<level>#{1,6})\s+(?P<text>.*?)\s*#*\s*$")
_FRONT_MATTER_LINE: Final = re.compile(r"^(?P<key>[A-Za-z0-9_-]+):\s*(?P<value>.*)$")

_COMMENT_START: Final = "<!--"
_COMMENT_END: Final = "-->"
_BLOCK_ELEMENTS: Final = re.compile(r"^\s*<(?P<tag>script|style)\b", re.IGNORECASE)
_PAGES_ONLY_START: Final = re.compile(r"^\s*<PagesOnly\s*>\s*$")
_PAGES_ONLY_END: Final = re.compile(r"^\s*</PagesOnly\s*>\s*$")
# A whole line holding one component tag: <Intro>, </Intro>, <InlineToc />, <div class="x">.
_TAG_LINE: Final = re.compile(r"^\s*</?(?:[A-Z][\w.]*|div|span)(?:\s[^<>]*)?/?>\s*$")
# The first line of a tag whose attributes continue on the next lines: <Image\n  alt="..."\n/>.
_TAG_START: Final = re.compile(r"^\s*<(?:[A-Z][\w.]*|div|span)(?:\s[^<>]*)?$")
_MDC_BLOCK: Final = re.compile(r"^\s*:{2,}(?P<name>[A-Za-z][\w-]*)?(?:\{[^}]*\})?\s*$")
_MDC_INLINE: Final = re.compile(r"^\s*(?::[A-Za-z][\w-]*(?:\{[^}]*\})?\s*)+$")
_MDC_SLOT: Final = re.compile(r"^\s*#[a-z][\w-]*\s*$")
_CONTAINER: Final = re.compile(r"^\s*:::\s*(?P<kind>[A-Za-z-]*)\s*(?P<title>.*?)\s*$")
_THEMATIC_BREAK: Final = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
_CALLOUT_NAMES: Final = frozenset({"note", "tip", "warning", "important", "caution", "danger"})

_MACRO: Final = re.compile(r"\{\{\s*(?P<name>[\w-]+)\s*(?:\((?P<args>.*?)\))?\s*\}\}")
_MACRO_ARGUMENT: Final = re.compile(r"""(?P<quote>["'])(?P<text>.*?)(?P=quote)""")
# Cross-reference macros render a link with a text; every other macro renders no prose.
_TEXT_MACROS: Final = frozenset(
    {
        "cssxref",
        "domxref",
        "glossary",
        "htmlattrxref",
        "htmlelement",
        "httpheader",
        "httpmethod",
        "httpstatus",
        "jsxref",
        "svgattr",
        "svgelement",
    }
)
# Link texts and image descriptions may hold one level of brackets: [[meta] Support](url).
_LINK_TEXT: Final = r"(?:[^\[\]]|\[[^\[\]]*\])+"
_LINK_TARGET: Final = r"\([^()\s]*(?:\([^)]*\)[^()\s]*)*(?:\s+\"[^\"]*\")?\)"
_IMAGE: Final = re.compile(rf"!\[{_LINK_TEXT}?\]{_LINK_TARGET}")
_LINK: Final = re.compile(rf"\[(?P<text>{_LINK_TEXT})\]{_LINK_TARGET}")
_INLINE_PAGES_ONLY: Final = re.compile(r"<PagesOnly\s*>.*?</PagesOnly\s*>")
_CLOSING_TAG: Final = re.compile(r"</([A-Za-z][\w.-]*)\s*>")
_TAG: Final = re.compile(
    r"<(?P<close>/)?(?P<name>[A-Za-z][\w.-]*)(?P<attributes>\s[^<>]*?)?\s*(?P<self>/)?>"
)
# Formatting and structure tags of the HTML that MDN embeds, for example in its tables.
_HTML_TAGS: Final = frozenset(
    {
        *("a", "abbr", "b", "blockquote", "br", "caption", "cite", "code", "col", "colgroup"),
        *("dd", "details", "div", "dl", "dt", "em", "figcaption", "figure", "hr", "i", "img"),
        *("kbd", "li", "mark", "ol", "p", "q", "samp", "section", "small", "span", "strong"),
        *("sub", "summary", "sup", "table", "tbody", "td", "tfoot", "th", "thead", "tr", "u"),
        *("ul", "var"),
    }
)
_CODE_SPAN: Final = re.compile(r"(`+).+?\1")
_HEADING_ANCHOR: Final = re.compile(r"\s*(?:\{/\*.*?\*/\}|\{#[^}]*\})\s*$")
_GITHUB_CALLOUT: Final = re.compile(r"^(?P<prefix>\s*>\s*)\[!(?P<kind>[A-Za-z]+)\]")
_DEFINITION: Final = re.compile(r"^(?P<indent>\s*)- : ")
_TWOSLASH_DIRECTIVE: Final = re.compile(r"^\s*//\s*(?:@\w+.*|\^\?.*|-{3}cut-{3}.*)$")
_BLANK_RUNS: Final = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class Section:
    """One section of a page.

    Attributes:
        path: The H2 heading and, inside it, the H3 heading; empty for the introduction.
        text: The cleaned Markdown below the heading, without the heading line.
    """

    path: tuple[str, ...]
    text: str


@dataclass(frozen=True)
class MarkdownPage:
    """A parsed page.

    Attributes:
        front_matter: The top-level scalar values of the front matter, as strings.
        title: The front matter's ``title``, else the first H1, else None; without
            backticks, because it is shown as plain text.
        sections: The sections with text, in page order.
    """

    front_matter: dict[str, str] = field(default_factory=dict)
    title: str | None = None
    sections: tuple[Section, ...] = ()


def parse_markdown(text: str) -> MarkdownPage:
    """Parse a Markdown or MDX page into its front matter, title and cleaned sections.

    Args:
        text: The page source.

    Returns:
        The parsed page (see the module docstring for the rules).
    """
    front_matter, body = split_front_matter(text)
    h1, sections = _split_sections(_clean(body))
    description = front_matter.get("description")
    if description:
        intro = next((s for s in sections if not s.path), None)
        if intro is None:
            sections.insert(0, Section(path=(), text=description))
        else:
            index = sections.index(intro)
            sections[index] = Section(path=(), text=f"{description}\n\n{intro.text}")
    title = front_matter.get("title") or h1
    return MarkdownPage(
        front_matter=front_matter,
        title=_plain(title) if title else None,
        sections=tuple(sections),
    )


def split_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Separate the YAML front matter from the page body.

    Args:
        text: The page source.

    Returns:
        The top-level scalar values of the front matter (quotes removed) and the body. A page
        without front matter, or with an unterminated block, returns ``({}, text)``.
    """
    lines = text.lstrip("﻿").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    end = next((i for i, line in enumerate(lines[1:], start=1) if line.strip() == "---"), None)
    if end is None:
        return {}, text
    values: dict[str, str] = {}
    for line in lines[1:end]:
        match = _FRONT_MATTER_LINE.match(line)
        if match is None:
            continue
        value = _unquote(match["value"].strip())
        if value and value not in {"|", ">", "|-", ">-"}:
            values[match["key"]] = value
    return values, "\n".join(lines[end + 1 :])


def _unquote(value: str) -> str:
    """Remove the quotes of a YAML scalar and undo its escapes."""
    if len(value) >= 2 and value[0] == value[-1] == '"':
        return value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    if len(value) >= 2 and value[0] == value[-1] == "'":
        return value[1:-1].replace("''", "'")
    return value


def _clean(body: str) -> str:
    """Apply the cleaning rules of the module docstring to the page body."""
    output: list[str] = []
    fence: str | None = None
    twoslash = False
    skip_until: re.Pattern[str] | str | None = None
    in_tag = False
    for line in body.splitlines():
        if fence is not None:
            stripped = line.strip()
            if stripped.startswith(fence) and set(stripped) <= {fence[0]}:
                fence = None
                output.append(line)
            elif not (twoslash and _TWOSLASH_DIRECTIVE.match(line)):
                output.append(line)
            continue
        if skip_until is not None:
            if isinstance(skip_until, str):
                if skip_until in line:
                    remainder = line.split(skip_until, 1)[1]
                    skip_until = None
                    if remainder.strip():
                        output.append(_inline(remainder))
            elif skip_until.search(line):
                skip_until = None
            continue
        if in_tag:
            in_tag = ">" not in line
            continue

        fence_match = _FENCE.match(line)
        if fence_match:
            fence = fence_match["fence"]
            twoslash = "twoslash" in fence_match["info"]
            output.append(line)
            continue
        if _COMMENT_START in line:
            line, skip_until = _strip_comments(line)
            if not line.strip():
                continue
        block = _BLOCK_ELEMENTS.match(line)
        if block:
            closing = re.compile(rf"</{block['tag']}\s*>", re.IGNORECASE)
            if not closing.search(line):
                skip_until = closing
            continue
        if _PAGES_ONLY_START.match(line):
            skip_until = _PAGES_ONLY_END
            continue
        if _TAG_LINE.match(line) or _MDC_INLINE.match(line) or _MDC_SLOT.match(line):
            continue
        if _TAG_START.match(line):
            in_tag = True
            continue
        if _THEMATIC_BREAK.match(line):
            continue
        mdc = _MDC_BLOCK.match(line)
        if mdc:
            name = (mdc["name"] or "").lower()
            if name in _CALLOUT_NAMES:
                output.append(f"**{name.capitalize()}:**")
            continue
        container = _CONTAINER.match(line)
        if container:
            if container["title"]:
                output.append(f"**{_inline(container['title'])}**")
            elif container["kind"].lower() in _CALLOUT_NAMES:
                output.append(f"**{container['kind'].capitalize()}:**")
            continue
        output.append(_inline(line))
    return "\n".join(output)


def _strip_comments(line: str) -> tuple[str, str | None]:
    """Remove the HTML comments of a line; report an unterminated one."""
    kept = []
    rest = line
    while _COMMENT_START in rest:
        before, after = rest.split(_COMMENT_START, 1)
        kept.append(before)
        if _COMMENT_END not in after:
            return "".join(kept), _COMMENT_END
        rest = after.split(_COMMENT_END, 1)[1]
    kept.append(rest)
    return "".join(kept), None


def _inline(line: str) -> str:
    """Apply the inline rules: images, links, macros, tags, anchors, callouts and definitions.

    Links and macros are replaced on the whole line, because a link's text is often a code
    span (``[`useState`](/reference/react/useState)``) and a macro's arguments may hold one.
    HTML and JSX tags are removed outside code spans only.
    """
    line = _IMAGE.sub("", line)
    line = _LINK.sub(r"\g<text>", line)
    if "{{" in line:
        line = _MACRO.sub(_render_macro, line)
        line = _outside_code_spans(line, lambda part: _MACRO.sub(_render_text_macro, part))
    if "<" in line:
        line = _INLINE_PAGES_ONLY.sub("", line)
        closing_tags = frozenset(_CLOSING_TAG.findall(line))
        line = _outside_code_spans(line, lambda part: _strip_tags(part, closing_tags))
    line = _HEADING_ANCHOR.sub("", line)
    line = _GITHUB_CALLOUT.sub(lambda m: f"{m['prefix']}**{m['kind'].capitalize()}:**", line)
    return _DEFINITION.sub(r"\g<indent>  ", line)


def _outside_code_spans(line: str, transform: Callable[[str], str]) -> str:
    """Apply ``transform`` to the parts of a line outside its code spans."""
    pieces: list[str] = []
    position = 0
    for span in _CODE_SPAN.finditer(line):
        pieces += [transform(line[position : span.start()]), span[0]]
        position = span.end()
    return "".join([*pieces, transform(line[position:])])


def _strip_tags(text: str, closing_tags: frozenset[str]) -> str:
    """Remove HTML formatting tags and JSX component tags, keeping the text between them.

    A capitalized name is removed only where it is clearly a component: a closing or
    self-closing tag, a tag with attributes, or an opening tag whose closing tag is on the
    same line (``closing_tags``, the names closed anywhere on the line). So TypeScript's
    ``Array<Type>`` in prose stays as it is.
    """

    def replace(match: re.Match[str]) -> str:
        name = match["name"]
        if name in _HTML_TAGS:
            return ""
        if name[0].isupper() and (
            match["close"] or match["self"] or match["attributes"] or name in closing_tags
        ):
            return ""
        return match[0]

    return _TAG.sub(replace, text)


def _render_macro(match: re.Match[str]) -> str:
    """Replace an MDN macro anywhere on a line: the display text of a cross-reference, or nothing.

    Only names that are clearly MDN's are replaced here: the cross-reference macros of
    :data:`_TEXT_MACROS`, capitalized names (``Compat``, ``EmbedLiveSample``), names with an
    underscore (``deprecated_inline``) and macros with arguments. A plain lower-case mustache
    such as Vue's ``{{ count }}`` may be template syntax in a code span, so it is replaced
    by a second pass that sees only the text outside code spans.
    """
    name = match["name"]
    if (
        name.lower() not in _TEXT_MACROS
        and not name[0].isupper()
        and "_" not in name
        and not match["args"]
    ):
        return match[0]
    return _render_text_macro(match)


def _render_text_macro(match: re.Match[str]) -> str:
    """Render a macro: the display text of a cross-reference macro, nothing for the others."""
    lowered = match["name"].lower()
    if lowered not in _TEXT_MACROS or not match["args"]:
        return ""
    arguments = [argument["text"] for argument in _MACRO_ARGUMENT.finditer(match["args"])]
    if not arguments:
        return ""
    if lowered == "htmlelement":
        text = f"<{arguments[0]}>"
    else:
        text = arguments[1] if len(arguments) > 1 and arguments[1] else arguments[0]
    in_code_span = match.string[match.start() - 1 : match.start()] == "`"
    return text if lowered == "glossary" or in_code_span else f"`{text}`"


def _split_sections(text: str) -> tuple[str | None, list[Section]]:
    """Split cleaned Markdown at its H2 and H3 headings (see the module docstring)."""
    h1: str | None = None
    h2: str | None = None
    path: tuple[str, ...] = ()
    lines: list[str] = []
    sections: list[Section] = []
    fence: str | None = None

    def close() -> None:
        body = _BLANK_RUNS.sub("\n\n", "\n".join(lines)).strip()
        if body:
            sections.append(Section(path=path, text=body))
        lines.clear()

    for line in text.splitlines():
        if fence is not None:
            stripped = line.strip()
            if stripped.startswith(fence) and set(stripped) <= {fence[0]}:
                fence = None
            lines.append(line)
            continue
        fence_match = _FENCE.match(line)
        if fence_match:
            fence = fence_match["fence"]
            lines.append(line)
            continue
        heading = _HEADING.match(line)
        if heading is None or len(heading["level"]) > 3:
            lines.append(line)
            continue
        level, title = len(heading["level"]), _plain(heading["text"])
        if level == 1:
            if h1 is None and not sections and not "\n".join(lines).strip():
                h1 = title
                continue
            level = 2
        close()
        if level == 2:
            h2 = title
            path = (title,)
        else:
            path = (h2, title) if h2 else (title,)
    close()
    return h1, sections


def _plain(text: str) -> str:
    """Turn a heading or title into plain text: no backticks, single spaces."""
    return " ".join(text.replace("`", "").split())
