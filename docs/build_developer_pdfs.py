"""Build the English and Hungarian developer guides from their Markdown sources.

Document tooling only; the application dependency files are intentionally independent.
Requires reportlab, and PyMuPDF/Pillow for the optional local visual QA.
Run from the repository root: python docs/build_developer_pdfs.py

The layout follows the PwC-inspired theme of the Streamlit UI (.streamlit/config.toml): the
same orange, rose and tangerine accents on warm greys, Georgia headings and Arial text. It
uses no logo or trademark.
"""

# The optional workspace-local dependency directory is added before third-party imports.
# ruff: noqa: E402

from __future__ import annotations

import re
import sys
from html import escape
from pathlib import Path
from typing import override

ROOT = Path(__file__).resolve().parents[1]
# A workspace-local install also works without modifying the project environment.
VENDOR = ROOT / "tmp/pdfs/vendor"
if VENDOR.is_dir():
    sys.path.insert(0, str(VENDOR))

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# The palette of the Streamlit theme (.streamlit/config.toml).
ORANGE_HEX = "#D04A02"
LINK_HEX = "#AD3517"
CODE_HEX = "#63351E"
ORANGE = colors.HexColor(ORANGE_HEX)
ROSE = colors.HexColor("#DB536A")
TANGERINE = colors.HexColor("#E8A630")
INK = colors.HexColor("#2D2D2D")
MUTED = colors.HexColor("#6E645E")
PALE = colors.HexColor("#F5F2EF")
RULE = colors.HexColor("#DDD5CE")
WIDTH, HEIGHT = A4
LEFT = 48
BODY_WIDTH = WIDTH - 96
COVER_SPLIT = 470  # lower edge of the cover's orange panel, in points from the bottom


def fonts():
    """Register a sans, a serif and a mono font with complete Hungarian glyph coverage."""
    aliases = ("Guide", "GuideBold", "GuideItalic", "GuideSerif", "GuideSerifBold", "GuideMono")
    candidates = [
        (
            Path("C:/Windows/Fonts"),
            (
                "arial.ttf",
                "arialbd.ttf",
                "ariali.ttf",
                "georgia.ttf",
                "georgiab.ttf",
                "consola.ttf",
            ),
        ),
        (
            Path("/usr/share/fonts/truetype/dejavu"),
            (
                "DejaVuSans.ttf",
                "DejaVuSans-Bold.ttf",
                "DejaVuSans-Oblique.ttf",
                "DejaVuSerif.ttf",
                "DejaVuSerif-Bold.ttf",
                "DejaVuSansMono.ttf",
            ),
        ),
    ]
    for directory, names in candidates:
        if all((directory / name).exists() for name in names):
            for alias, name in zip(aliases, names, strict=True):
                pdfmetrics.registerFont(TTFont(alias, str(directory / name)))
            pdfmetrics.registerFontFamily(
                "Guide",
                normal="Guide",
                bold="GuideBold",
                italic="GuideItalic",
                boldItalic="GuideBold",
            )
            pdfmetrics.registerFontFamily(
                "GuideSerif",
                normal="GuideSerif",
                bold="GuideSerifBold",
                italic="GuideSerif",
                boldItalic="GuideSerifBold",
            )
            return
    raise RuntimeError(
        "Install Arial, Georgia and Consolas, or DejaVu Sans, Serif and Sans Mono, "
        "for PDF generation."
    )


fonts()
STYLES = {
    "body": ParagraphStyle(
        "body",
        fontName="Guide",
        fontSize=10,
        leading=14.3,
        textColor=INK,
        spaceAfter=8,
        splitLongWords=True,
    ),
    "h1": ParagraphStyle(
        "h1",
        fontName="GuideSerif",
        fontSize=23,
        leading=28,
        textColor=INK,
        spaceAfter=17,
        keepWithNext=True,
    ),
    "h2": ParagraphStyle(
        "h2",
        fontName="GuideBold",
        fontSize=12.4,
        leading=17,
        textColor=ORANGE,
        spaceBefore=11,
        spaceAfter=7,
        keepWithNext=True,
    ),
    "bullet": ParagraphStyle(
        "bullet",
        fontName="Guide",
        fontSize=10,
        leading=14.3,
        textColor=INK,
        leftIndent=11,
        firstLineIndent=-10,
        spaceAfter=6,
    ),
    "note": ParagraphStyle(
        "note", fontName="Guide", fontSize=9.2, leading=13, textColor=MUTED, spaceAfter=9
    ),
    "callout": ParagraphStyle(
        "callout", fontName="Guide", fontSize=9.2, leading=13, textColor=INK, spaceAfter=0
    ),
    "code": ParagraphStyle(
        "code", fontName="GuideMono", fontSize=8.25, leading=11.5, textColor=INK, spaceAfter=0
    ),
    "cell": ParagraphStyle(
        "cell", fontName="Guide", fontSize=9, leading=12.4, textColor=INK, spaceAfter=0
    ),
    "headcell": ParagraphStyle(
        "headcell",
        fontName="GuideBold",
        fontSize=9,
        leading=12.4,
        textColor=colors.white,
        spaceAfter=0,
    ),
    "toc": ParagraphStyle(
        "toc", fontName="Guide", fontSize=10, leading=13.8, textColor=INK, spaceAfter=0
    ),
}
COVER_STYLES = {
    "kicker": ParagraphStyle(
        "cover-kicker", fontName="GuideBold", fontSize=9.5, leading=13, textColor=colors.white
    ),
    "title": ParagraphStyle(
        "cover-title", fontName="GuideSerif", fontSize=44, leading=50, textColor=colors.white
    ),
    "subtitle": ParagraphStyle(
        "cover-subtitle", fontName="GuideSerif", fontSize=22, leading=27, textColor=colors.white
    ),
    "desc": ParagraphStyle(
        "cover-desc", fontName="Guide", fontSize=10.5, leading=15, textColor=colors.white
    ),
    "meta": ParagraphStyle(
        "cover-meta", fontName="GuideBold", fontSize=10, leading=14.3, textColor=INK
    ),
    "note": ParagraphStyle(
        "cover-note", fontName="Guide", fontSize=9.2, leading=13, textColor=MUTED
    ),
}


def inline(text: str) -> str:
    """Render a deliberately small, explicit Markdown subset."""
    pieces = re.split(r"(`[^`]+`)", text)
    result = []
    for piece in pieces:
        if piece.startswith("`") and piece.endswith("`"):
            result.append(
                f'<font name="GuideMono" size="8.6" color="{CODE_HEX}">'
                + escape(piece[1:-1])
                + "</font>"
            )
        else:
            piece = escape(piece)
            piece = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", piece)
            piece = re.sub(
                r"\[([^\]]+)\]\((https?://[^)]+)\)",
                rf'<link href="\2" color="{LINK_HEX}"><u>\1</u></link>',
                piece,
            )
            result.append(piece)
    return "".join(result)


def paragraph(text: str, style="body"):
    """Create a paragraph using the guide's inline formatter."""
    return Paragraph(inline(text), STYLES[style])


def heading(text: str, style="h1"):
    """Set a chapter heading such as ``01 / Scope`` with its number in the accent colour."""
    match = re.fullmatch(r"(\d+) / (.+)", text)
    if match is None:
        return paragraph(text, style)
    number, title = match.groups()
    return Paragraph(
        f'<font color="{ORANGE_HEX}">{number}</font>&#160;&#160;{inline(title)}', STYLES[style]
    )


class Diagram(Flowable):
    """A compact vector architecture diagram with selectable text."""

    def __init__(self, lang):
        """Select labels for one language edition."""
        super().__init__()
        self.lang = lang
        self.width = BODY_WIDTH
        self.height = 235

    def draw(self):
        """Draw calls with the Ollama connection outside the deterministic tool box."""
        c = self.canv
        labels = {
            "en": [
                ("Streamlit UI", "conversation + live steps"),
                ("LangGraph agent", "route / plan / synthesize / verify"),
                ("RAG subgraph", "rewrite / retrieve / grade / context"),
                ("Exact tools", "contrast / specificity / support"),
                ("Chroma + SQLite FTS5", "vectors + BM25"),
                ("Local Ollama", "qwen3.5:4b"),
            ],
            "hu": [
                ("Streamlit felület", "beszélgetés + élő lépések"),
                ("LangGraph ügynök", "irányítás / tervezés / ellenőrzés"),
                ("RAG-részgráf", "átírás / keresés / szűrés / kontextus"),
                ("Determin. eszközök", "kontraszt / specificitás / támogatás"),
                ("Chroma + SQLite FTS5", "vektorok + BM25"),
                ("Helyi Ollama", "qwen3.5:4b"),
            ],
        }[self.lang]
        boxes = [
            (135, 189, 230, 43),
            (135, 127, 230, 43),
            (0, 62, 241, 43),
            (258, 62, 225, 43),
            (0, 0, 241, 43),
            (258, 0, 225, 43),
        ]
        c.setStrokeColor(ORANGE)
        c.setFillColor(ORANGE)
        c.setLineWidth(1)
        for x1, y1, x2, y2 in [
            (250, 189, 250, 170),
            (190, 127, 121, 105),
            (310, 127, 370, 105),
            (121, 62, 121, 43),
        ]:
            c.line(x1, y1, x2, y2)
            c.circle(x2, y2, 2.0, fill=1, stroke=0)
        c.line(365, 149, 495, 149)
        c.line(495, 149, 495, 21)
        c.line(495, 21, 483, 21)
        c.circle(483, 21, 2.0, fill=1, stroke=0)
        for index, ((x, y, w, h), (title, desc)) in enumerate(zip(boxes, labels, strict=True)):
            # The agent, the centre of the system, is the one box set in the accent colour.
            agent = index == 1
            c.setFillColor(ORANGE if agent else PALE)
            c.setStrokeColor(ORANGE if agent else RULE)
            c.roundRect(x, y, w, h, 4, fill=1, stroke=1)
            c.setFillColor(colors.white if agent else INK)
            c.setFont("GuideBold", 10.4)
            c.drawCentredString(x + w / 2, y + 26, title)
            c.setFillColor(colors.white if agent else MUTED)
            c.setFont("Guide", 8.5)
            c.drawCentredString(x + w / 2, y + 11, desc)


def code_block(lines):
    """Wrap long commands visibly without changing the stored Markdown example."""
    rendered = []
    for line in lines:
        if not line:
            rendered.append("&#160;")
            continue
        # A smaller font for a few long commands keeps copy/paste reliable.
        rendered.append(escape(line).replace(" ", "&#160;"))
    longest = max((pdfmetrics.stringWidth(line, "GuideMono", 8.25) for line in lines), default=0)
    style = STYLES["code"]
    if longest > BODY_WIDTH - 24:
        style = ParagraphStyle(
            "compact-code",
            parent=style,
            fontSize=max(6.5, 8.25 * (BODY_WIDTH - 24) / longest),
            leading=11.5,
        )
    block = Paragraph("<br/>".join(rendered), style)
    # Space after a table, unlike a trailing Spacer, never moves to a page of its own.
    table = Table([[block]], colWidths=[BODY_WIDTH], spaceAfter=10)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE),
                ("LINEBEFORE", (0, 0), (0, -1), 2.5, ORANGE),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ]
        )
    )
    return [KeepTogether([table])]


def callout(text):
    """Set a Markdown quote as a note with an accent bar."""
    table = Table([[paragraph(text, "callout")]], colWidths=[BODY_WIDTH], spaceAfter=9)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE),
                ("LINEBEFORE", (0, 0), (0, -1), 2.5, TANGERINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return [table]


def md_table(rows):
    """Render wrapped, shaded table rows."""
    cells = [
        [paragraph(value, "headcell" if n == 0 else "cell") for value in row]
        for n, row in enumerate(rows)
    ]
    ncols = len(rows[0])
    if ncols == 2:
        widths = [BODY_WIDTH * 0.34, BODY_WIDTH * 0.66]
    elif ncols == 3:
        widths = [BODY_WIDTH * 0.31, BODY_WIDTH * 0.21, BODY_WIDTH * 0.48]
    else:
        widths = [BODY_WIDTH / ncols] * ncols
    table = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT", spaceAfter=10)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), INK),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("LINEBELOW", (0, 0), (-1, 0), 2, ORANGE),
                ("LINEBELOW", (0, -1), (-1, -1), 0.5, RULE),
            ]
        )
    )
    return [table]


def parse_page(text, lang):
    """Parse one deliberately paginated Markdown chapter."""
    lines = text.strip().splitlines()
    story = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line.startswith("```"):
            index += 1
            block = []
            while index < len(lines) and not lines[index].startswith("```"):
                block.append(lines[index])
                index += 1
            story.extend(code_block(block))
        elif line.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                row = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[:\s-]+", cell) for cell in row):
                    rows.append(row)
                index += 1
            story.extend(md_table(rows))
            continue
        elif line == "<!-- architecture -->":
            story.extend([Diagram(lang), Spacer(1, 16)])
        elif line.startswith("# "):
            chapter = heading(line[2:])
            chapter.bookmark = line[2:]
            story.append(chapter)
        elif line.startswith("## "):
            story.append(paragraph(line[3:], "h2"))
        elif line.startswith("> "):
            story.extend(callout(line[2:]))
        elif line.startswith("- "):
            story.append(
                Paragraph(
                    f'<font color="{ORANGE_HEX}">•</font> ' + inline(line[2:]), STYLES["bullet"]
                )
            )
        elif match := re.match(r"^(\d+\.) (.*)$", line):
            number, item = match.groups()
            story.append(
                Paragraph(
                    f'<font color="{ORANGE_HEX}"><b>{number}</b></font> {inline(item)}',
                    STYLES["bullet"],
                )
            )
        else:
            group = [line]
            while (
                index + 1 < len(lines)
                and lines[index + 1].strip()
                and not re.match(r"^(#|\||>|-|```|<!--|\d+\. )", lines[index + 1])
            ):
                index += 1
                group.append(lines[index].strip())
            story.append(paragraph(" ".join(group)))
        index += 1
    return story


class GuideDoc(BaseDocTemplate):
    """Add a PDF outline for every numbered section."""

    @override
    def afterFlowable(self, flowable):
        """Bookmark each completed chapter heading."""
        if getattr(flowable, "bookmark", None):
            key = "chapter-" + str(self.page)
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(flowable.bookmark, key, 0, False)


def place(c, text, style, top, width=BODY_WIDTH):
    """Draw a wrapped paragraph whose first line starts at ``top``; return its lower edge."""
    block = Paragraph(text, style)
    _, height = block.wrapOn(c, width, HEIGHT)
    block.drawOn(c, LEFT, top - height)
    return top - height


def draw_cover(c, lang, title):
    """Draw the cover: an orange title panel with layered accent squares on its edge."""
    en = lang == "en"
    c.setFillColor(ORANGE)
    c.rect(0, COVER_SPLIT, WIDTH, HEIGHT - COVER_SPLIT, stroke=0, fill=1)
    c.setFillAlpha(0.92)
    c.setFillColor(ROSE)
    c.rect(WIDTH - 178, COVER_SPLIT - 58, 112, 112, stroke=0, fill=1)
    c.setFillColor(TANGERINE)
    c.rect(WIDTH - 96, COVER_SPLIT - 128, 96, 96, stroke=0, fill=1)
    c.setFillAlpha(1)

    kicker = "ENGINEERING DOCUMENTATION" if en else "MŰSZAKI DOKUMENTÁCIÓ"
    desc = (
        "A practical guide to developing, running and extending the local frontend "
        "documentation assistant."
        if en
        else "Gyakorlati útmutató a helyi frontend-dokumentációs asszisztens "
        "fejlesztéséhez, futtatásához és bővítéséhez."
    )
    top = place(c, kicker, COVER_STYLES["kicker"], HEIGHT - 96) - 22
    top = place(c, "Agentic RAG<br/>Chatbot", COVER_STYLES["title"], top) - 16
    top = place(c, escape(title), COVER_STYLES["subtitle"], top) - 12
    place(c, escape(desc), COVER_STYLES["desc"], top, BODY_WIDTH * 0.8)

    top = (
        place(
            c,
            "Python 3.12 / LangGraph / Streamlit / Ollama / Chroma",
            COVER_STYLES["meta"],
            COVER_SPLIT - 46,
            BODY_WIDTH * 0.8,
        )
        - 4
    )
    version = "Proof of concept · version 0.1.0" if en else "Működő prototípus · 0.1.0 verzió"
    place(c, version, COVER_STYLES["note"], top, BODY_WIDTH * 0.8)

    status = (
        "Covers the RAG reliability update and its measurements of 4 October 2026; the "
        "measurements of 3 October 2026 serve as the baseline."
        if en
        else "A RAG-megbízhatósági frissítést és 2026. október 4-i méréseit tartalmazza; "
        "a 2026. október 3-i mérések az alapértéket adják."
    )
    c.setFillColor(ORANGE)
    c.rect(LEFT, 168, 28, 2.5, stroke=0, fill=1)
    top = place(c, escape(status), COVER_STYLES["note"], 156, BODY_WIDTH * 0.8) - 6
    place(c, "4 October 2026" if en else "2026. október 4.", COVER_STYLES["note"], top)


def build(lang):
    """Write a single language edition with outline and page furniture."""
    source = ROOT / f"docs/developer-guide.{lang}.md"
    pages = source.read_text(encoding="utf-8").split("<!-- pagebreak -->")
    output = ROOT / f"output/pdf/developer-guide-{lang}.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    expected_pages = len(pages) + 2
    title = "Developer guide" if lang == "en" else "Fejlesztői dokumentáció"
    meta = "0.1.0  |  2026-10-04  |  " + lang.upper()

    def on_page(c, doc):
        c.saveState()
        c.setLineWidth(0.6)
        if doc.page == 1:
            draw_cover(c, lang, title)
            c.setFont("Guide", 8)
            c.setFillColor(MUTED)
            c.drawString(LEFT, 27, meta)
            c.restoreState()
            return
        c.setFillColor(ORANGE)
        c.rect(LEFT, HEIGHT - 33, 16, 3, stroke=0, fill=1)
        c.setFillColor(MUTED)
        c.setFont("Guide", 7.5)
        c.drawString(LEFT + 24, HEIGHT - 34, "AGENTIC RAG CHATBOT  ·  " + title.upper())
        c.setStrokeColor(RULE)
        c.line(LEFT, HEIGHT - 42, WIDTH - LEFT, HEIGHT - 42)
        c.line(LEFT, 41, WIDTH - LEFT, 41)
        c.setFont("Guide", 8)
        c.drawString(LEFT, 27, meta)
        total = f" / {expected_pages}"
        c.drawRightString(WIDTH - LEFT, 27, total)
        c.setFont("GuideBold", 8)
        c.setFillColor(ORANGE)
        c.drawRightString(
            WIDTH - LEFT - pdfmetrics.stringWidth(total, "Guide", 8), 27, str(doc.page)
        )
        c.restoreState()

    doc = GuideDoc(
        str(output),
        pagesize=A4,
        leftMargin=LEFT,
        rightMargin=LEFT,
        topMargin=62,
        bottomMargin=54,
        title=f"Agentic RAG Chatbot - {title}",
        author="Csaba Ovari",
        subject="Repository-based developer onboarding, architecture and operations",
        pageCompression=1,
    )
    doc.addPageTemplates(
        [
            PageTemplate(
                id="guide",
                frames=Frame(
                    LEFT,
                    54,
                    BODY_WIDTH,
                    HEIGHT - 116,
                    leftPadding=0,
                    rightPadding=0,
                    topPadding=0,
                    bottomPadding=0,
                ),
                onPage=on_page,
            )
        ]
    )
    # The cover is drawn by on_page; its page holds no flowables.
    story = [Spacer(1, 1), PageBreak()]
    story.append(paragraph("Contents" if lang == "en" else "Tartalomjegyzék", "h1"))
    toc = []
    for index, page in enumerate(pages):
        chapter = re.search(r"^# (.+)$", page, re.M).group(1)
        toc.append([heading(chapter, "toc"), paragraph(str(index + 3), "toc")])
    table = Table(toc, colWidths=[BODY_WIDTH - 38, 38], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("ALIGN", (1, 0), (1, -1), "RIGHT"),
                ("LINEBELOW", (0, 0), (-1, -1), 0.35, RULE),
            ]
        )
    )
    story.extend([table, Spacer(1, 18)])
    note = (
        "Commands run from the repository root. Shell-specific examples are labelled. "
        "Code identifiers, environment variables and file paths are identical "
        "in both language editions."
        if lang == "en"
        else "A parancsokat a projekt gyökérkönyvtárából kell futtatni. A példák jelölik "
        "a használt parancsértelmezőt. A kódazonosítók, környezeti változók és "
        "fájlútvonalak mindkét nyelvi változatban azonosak."
    )
    story.append(paragraph(note, "note"))
    for page in pages:
        story.append(PageBreak())
        story.extend(parse_page(page, lang))
    doc.build(story)
    if doc.page != expected_pages:
        raise RuntimeError(
            f"Pagination changed: expected {expected_pages}, got {doc.page}. "
            "Split or shorten the overflowing chapter before delivery."
        )
    print(f"Created {output.relative_to(ROOT)} ({expected_pages} planned pages)")


if __name__ == "__main__":
    for language in ("en", "hu"):
        build(language)
