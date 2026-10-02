"""Tests for the agent's tools: contrast, specificity, browser support and the tool layer.

The pure modules (``contrast``, ``specificity``, ``compat``) are checked against published
values: the WCAG formula, the examples of the Selectors Level 4 specification, and a small
browser-compat-data fixture written under ``tmp_path`` in BCD's own layout, ``mirror``
statements included. The LangChain tools are checked through ``invoke``, as ``call_tool``
calls them.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import ToolException
from pydantic import ValidationError

from agentic_rag.agent.compat import BCD_DIRECTORY, CompatData, parse_target
from agentic_rag.agent.contrast import contrast_ratio, evaluate_contrast, parse_color
from agentic_rag.agent.specificity import (
    Specificity,
    compare_specificity,
    specificity,
    split_selector_list,
)
from agentic_rag.agent.tools import (
    BROWSER_SUPPORT,
    CHECK_CONTRAST,
    CSS_SPECIFICITY,
    SEARCH_KNOWLEDGE_BASE,
    BrowserSupportTool,
    CheckContrastTool,
    CssSpecificityTool,
    SearchKnowledgeBaseTool,
    get_non_retrieval_tools,
    get_tools,
)
from agentic_rag.config import Settings
from agentic_rag.rag.state import RagInput, RagOutput, Source

# --- contrast ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("foreground", "background", "ratio"),
    [
        ("#000", "#fff", 21.0),
        ("white", "white", 1.0),
        ("#777777", "#ffffff", 4.47),  # 4.478..., rounded down: just below AA
        ("#767676", "#fff", 4.54),  # the lightest grey that passes AA on white
        ("rgb(119, 119, 119)", "white", 4.47),
        ("rgb(119 119 119 / 100%)", "#FFF", 4.47),
        ("hsl(0 0% 46.67%)", "#fff", 4.47),
        ("navy", "yellow", 14.9),
    ],
)
def test_contrast_ratios_follow_the_wcag_formula(
    foreground: str, background: str, ratio: float
) -> None:
    assert evaluate_contrast(foreground, background).ratio == ratio


def test_the_ratio_does_not_depend_on_the_order_of_opaque_colours() -> None:
    dark, light = parse_color("#333"), parse_color("#eee")

    assert contrast_ratio(dark, light) == contrast_ratio(light, dark)


def test_a_translucent_foreground_is_painted_over_the_background() -> None:
    result = evaluate_contrast("rgba(0, 0, 0, 0.5)", "#ffffff")

    # Half black over white is 127.5 per channel: the ratio uses the exact value, the hex the
    # nearest one.
    assert result.foreground.hex == "#808080"
    assert result.ratio == evaluate_contrast("rgb(127.5 127.5 127.5)", "#fff").ratio == 3.97


def test_verdicts_cover_aa_aaa_and_non_text_contrast() -> None:
    verdicts = {
        (level, what): passes
        for level, what, _, passes in evaluate_contrast("#777", "#fff").verdicts
    }

    assert verdicts == {
        ("AA", "normal text"): False,
        ("AA", "large text"): True,
        ("AAA", "normal text"): False,
        ("AAA", "large text"): False,
        ("AA", "UI components and graphics"): True,
    }


@pytest.mark.parametrize(
    ("foreground", "background", "problem"),
    [
        ("#77", "#fff", "unsupported colour"),
        ("blurple", "#fff", "unsupported colour"),
        ("rgb(300, 0, 0)", "#fff", "out of range"),
        ("rgb(1, 2)", "#fff", "cannot read"),
        ("#000", "rgba(255, 255, 255, 0.5)", "must be opaque"),
    ],
)
def test_unreadable_colours_are_rejected(foreground: str, background: str, problem: str) -> None:
    with pytest.raises(ValueError, match=problem):
        evaluate_contrast(foreground, background)


# --- specificity ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        # The examples of Selectors Level 4, section 17.
        ("*", (0, 0, 0)),
        ("LI", (0, 0, 1)),
        ("UL LI", (0, 0, 2)),
        ("UL OL+LI", (0, 0, 3)),
        ("H1 + *[REL=up]", (0, 1, 1)),
        ("UL OL LI.red", (0, 1, 3)),
        ("LI.red.level", (0, 2, 1)),
        ("#x34y", (1, 0, 0)),
        ("#s12:not(FOO)", (1, 0, 1)),
        (".foo :is(.bar, #baz)", (1, 1, 0)),
        # Further rules of the module docstring.
        (":where(#a .b) p", (0, 0, 1)),
        ("a:hover::after", (0, 1, 2)),
        ("p:before", (0, 0, 2)),
        (":nth-child(2n+1 of .item.active)", (0, 3, 0)),
        (":nth-child(2n+1)", (0, 1, 0)),
        (":has(> img, #hero)", (1, 0, 0)),
        ("svg|circle", (0, 0, 1)),
        ("*|*", (0, 0, 0)),
        ('[data-x="a,b]"] .y', (0, 2, 0)),
        ("::slotted(span.x)", (0, 1, 2)),
        (":host(.dark) a", (0, 2, 1)),
        ("input:not([type=checkbox]):focus-visible", (0, 2, 1)),
    ],
)
def test_specificity_follows_selectors_level_4(
    selector: str, expected: tuple[int, int, int]
) -> None:
    assert specificity(selector) == Specificity(*expected)


def test_selector_lists_are_split_at_top_level_commas() -> None:
    assert split_selector_list("a, :is(b, c), [x='1,2'] ,") == ["a", ":is(b, c)", "[x='1,2']"]
    assert [str(value) for _, value in compare_specificity(["a, #b"])] == ["(0, 0, 1)", "(1, 0, 0)"]


@pytest.mark.parametrize(
    ("selector", "problem"),
    [
        ("", "empty"),
        ("a, b", "selector list"),
        ("& .child", "nesting selector"),
        ("a:is(b", "unclosed"),
        ("a % b", "unexpected"),
    ],
)
def test_invalid_selectors_are_rejected(selector: str, problem: str) -> None:
    with pytest.raises(ValueError, match=problem):
        specificity(selector)


# --- browser support -----------------------------------------------------------------------


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write one BCD-style JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def release(version: str, engine: str, engine_version: str) -> tuple[str, dict[str, str]]:
    """One entry of a browser's ``releases``."""
    return version, {"engine": engine, "engine_version": engine_version}


def browser(
    key: str, *releases: tuple[str, dict[str, str]], upstream: str | None = None
) -> dict[str, Any]:
    """A ``browsers/<key>.json`` document."""
    info: dict[str, Any] = {"name": key, "releases": dict(releases)}
    if upstream:
        info["upstream"] = upstream
    return {"browsers": {key: info}}


@pytest.fixture
def bcd(tmp_path: Path) -> CompatData:
    """A tiny browser-compat-data download with mirrored browsers."""
    root = tmp_path / BCD_DIRECTORY
    blink = [release(str(version), "Blink", str(version)) for version in (37, 79, 104, 105, 110)]
    write_json(root / "browsers" / "chrome.json", browser("chrome", *blink))
    write_json(
        root / "browsers" / "edge.json",
        browser("edge", release("18", "EdgeHTML", "18"), *blink[1:], upstream="chrome"),
    )
    write_json(
        root / "browsers" / "safari.json",
        browser("safari", release("15", "WebKit", "612.1"), release("15.4", "WebKit", "613.1")),
    )
    write_json(
        root / "browsers" / "safari_ios.json",
        browser(
            "safari_ios",
            release("15", "WebKit", "612.1"),
            release("15.4", "WebKit", "613.1"),
            upstream="safari",
        ),
    )
    has = {
        "__compat": {
            "mdn_url": "https://developer.mozilla.org/docs/Web/CSS/:has",
            "support": {
                "chrome": {"version_added": "105"},
                "edge": "mirror",
                "firefox": [{"version_added": "121"}, {"version_added": "103", "flags": [{}]}],
                "safari": {"version_added": "15.4"},
                "safari_ios": "mirror",
            },
            "status": {"experimental": False, "standard_track": True, "deprecated": False},
        }
    }
    write_json(root / "css" / "selectors" / "has.json", {"css": {"selectors": {"has": has}}})
    dialog = {"__compat": {"support": {"chrome": {"version_added": "37"}, "edge": "mirror"}}}
    write_json(
        root / "html" / "elements" / "dialog.json", {"html": {"elements": {"dialog": dialog}}}
    )
    has_api = {"__compat": {"support": {"chrome": {"version_added": "≤79"}}}}
    write_json(root / "api" / "Headers.json", {"api": {"Headers": {"has": has_api}}})
    old = {
        "__compat": {
            "support": {"chrome": {"version_added": "1", "version_removed": "104"}},
            "status": {"deprecated": True, "standard_track": False},
        }
    }
    write_json(
        root / "css" / "properties" / "zoom-old.json", {"css": {"properties": {"zoom-old": old}}}
    )
    return CompatData(root)


@pytest.mark.parametrize(
    ("name", "area", "first"),
    [
        (":has()", None, "css.selectors.has"),
        ("`:has`", None, "css.selectors.has"),
        ("css.selectors.has", None, "css.selectors.has"),
        ("<dialog>", None, "html.elements.dialog"),
        ("dialog", "html", "html.elements.dialog"),
        ("has", "api", "api.Headers.has"),
        ("Headers.has", None, "api.Headers.has"),
    ],
)
def test_features_are_found_by_the_names_people_write(
    bcd: CompatData, name: str, area: Any, first: str
) -> None:
    assert bcd.find(name, area)[0].id == first


def test_a_bare_name_lists_every_match_shortest_first(bcd: CompatData) -> None:
    assert [feature.id for feature in bcd.find("has")] == ["api.Headers.has", "css.selectors.has"]
    assert bcd.find("useState") == []


def test_mirror_statements_follow_the_upstream_engine(bcd: CompatData) -> None:
    has = bcd.find(":has()")[0]
    dialog = bcd.find("<dialog>")[0]

    assert bcd.summary(has) == [
        "Chrome: since 105",
        "Edge: since 105",
        "Firefox: since 121",
        "Safari: since 15.4",
        "Chrome Android: no data",
        "Safari on iOS: since 15.4",
    ]
    # Chrome 37 predates the first Blink-based Edge, so Edge supports it from 79.
    assert bcd.summary(dialog)[1] == "Edge: since 79"


@pytest.mark.parametrize(
    ("target", "supported", "detail"),
    [
        ("Safari 15", False, "added in 15.4"),
        ("Safari 15.4", True, "added in 15.4"),
        ("iOS Safari 16", True, "added in 15.4"),
        ("Edge 104", False, "added in 105"),
        ("chrome>=110", True, "added in 105"),
        ("Firefox 120", False, "added in 121"),
        ("Opera 100", None, "no data"),
    ],
)
def test_target_versions_are_checked(
    bcd: CompatData, target: str, supported: bool | None, detail: str
) -> None:
    verdict = bcd.check(bcd.find(":has()")[0], *parse_target(target))

    assert (verdict.supported, verdict.detail) == (supported, detail)


def test_ranges_and_removals_are_respected(bcd: CompatData) -> None:
    ranged = bcd.find("Headers.has")[0]
    removed = bcd.find("zoom-old")[0]

    assert bcd.check(ranged, "chrome", "79").supported is True
    assert bcd.check(ranged, "chrome", "60").supported is None
    assert bcd.check(removed, "chrome", "103").supported is True
    assert bcd.check(removed, "chrome", "104").detail == "removed in 104"


@pytest.mark.parametrize(
    ("text", "problem"), [("Safari", "cannot read"), ("Netscape 4", "unknown browser")]
)
def test_unreadable_targets_are_rejected(text: str, problem: str) -> None:
    with pytest.raises(ValueError, match=problem):
        parse_target(text)


def test_missing_data_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="ingest --download"):
        CompatData(tmp_path / "missing").find(":has()")


# --- the LangChain tools -------------------------------------------------------------------


def test_check_contrast_describes_every_verdict() -> None:
    output = CheckContrastTool().invoke({"foreground": "#777777", "background": "white"})

    assert output.splitlines()[0] == (
        "Contrast ratio 4.47:1 for #777777 on #ffffff (WCAG 2.2, rounded down)."
    )
    assert "- AA normal text (at least 4.5:1): fails" in output
    assert "- AA large text (at least 3:1): passes" in output


def test_css_specificity_names_the_winner_and_ties() -> None:
    tool = CssSpecificityTool()

    winner = tool.invoke({"selectors": ["#nav .item", ".menu a:hover"]})
    tie = tool.invoke({"selectors": [".a .b", ".c.d"]})

    assert "- `#nav .item`: (1, 1, 0)" in winner
    assert "`#nav .item` has the highest specificity (1, 1, 0) and wins." in winner
    assert "`.a .b`, `.c.d` share the highest specificity (0, 2, 0)" in tie


def test_browser_support_summarizes_and_checks_targets(bcd: CompatData) -> None:
    output = BrowserSupportTool(data=bcd).invoke({"feature": ":has()", "browsers": ["Safari 15"]})

    assert (
        output.splitlines()[0] == "Browser support of css.selectors.has (MDN browser-compat-data):"
    )
    assert "- Safari: since 15.4" in output
    assert "Safari 15: not supported (added in 15.4)." in output
    assert "MDN: https://developer.mozilla.org/docs/Web/CSS/:has" in output


def test_browser_support_reports_status_and_alternatives(bcd: CompatData) -> None:
    removed = BrowserSupportTool(data=bcd).invoke({"feature": "zoom-old"})
    ambiguous = BrowserSupportTool(data=bcd).invoke({"feature": "has"})

    assert "Status: deprecated, non-standard." in removed
    assert "Other features with this name: css.selectors.has." in ambiguous


@pytest.mark.parametrize(
    ("tool_input", "problem"),
    [
        ({"feature": "useState"}, "No feature named 'useState'"),
        ({"feature": ":has()", "browsers": ["Netscape 4"]}, "unknown browser"),
    ],
)
def test_browser_support_raises_tool_exceptions(
    bcd: CompatData, tool_input: dict[str, Any], problem: str
) -> None:
    with pytest.raises(ToolException, match=problem):
        BrowserSupportTool(data=bcd).invoke(tool_input)


@pytest.mark.parametrize(
    ("tool", "tool_input"),
    [
        (CheckContrastTool(), {"foreground": "#12", "background": "#fff"}),
        (CssSpecificityTool(), {"selectors": ["& p"]}),
    ],
)
def test_bad_input_raises_a_tool_exception(tool: Any, tool_input: dict[str, Any]) -> None:
    with pytest.raises(ToolException):
        tool.invoke(tool_input)


@pytest.mark.parametrize(
    ("tool", "tool_input"),
    [
        (CheckContrastTool(), {"foreground": "#777"}),
        (CheckContrastTool(), {"foreground": "#777", "background": "#fff", "size": 12}),
        (CssSpecificityTool(), {"selectors": []}),
        (CssSpecificityTool(), {"selectors": ["a"] * 11}),
    ],
)
def test_arguments_are_validated_before_the_tool_runs(
    tool: Any, tool_input: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError):
        tool.invoke(tool_input)


def stand_in_rag_graph(rag_input: RagInput) -> RagOutput:
    """A RAG subgraph that finds one chunk for every query."""
    source = Source(chunk_id="c1", source="react/useState.md", content="useState adds state.")
    return RagOutput(context=f"[1] {rag_input['query']}", sources=[source], trace=[])


def test_the_search_tool_returns_the_context_and_the_whole_output() -> None:
    tool = SearchKnowledgeBaseTool(rag_graph=RunnableLambda(stand_in_rag_graph))

    message = tool.invoke(
        {"type": "tool_call", "id": "call_1", "name": SEARCH_KNOWLEDGE_BASE, "args": {"query": "q"}}
    )

    assert isinstance(message, ToolMessage)
    assert message.content == "[1] q"
    assert message.artifact["sources"][0].chunk_id == "c1"


def test_the_search_tool_has_its_final_name_schema_and_response_format() -> None:
    tool = SearchKnowledgeBaseTool(rag_graph=RunnableLambda(stand_in_rag_graph))

    assert tool.name == SEARCH_KNOWLEDGE_BASE == "search_knowledge_base"
    assert tool.response_format == "content_and_artifact"
    schema = tool.tool_call_schema.model_json_schema()
    assert set(schema["properties"]) == {"query"}
    assert schema["required"] == ["query"]
    assert "rag_graph" not in tool.model_dump()
    with pytest.raises(ValidationError):
        tool.invoke({"query": ""})
    with pytest.raises(ValidationError):
        SearchKnowledgeBaseTool.model_validate({"rag_graph": "not a runnable"})


def test_the_tool_set_has_the_search_tool_first_and_unique_names(settings: Settings) -> None:
    tools = get_tools(settings, rag_graph=RunnableLambda(stand_in_rag_graph))

    assert [tool.name for tool in tools] == [
        SEARCH_KNOWLEDGE_BASE,
        CHECK_CONTRAST,
        CSS_SPECIFICITY,
        BROWSER_SUPPORT,
    ]
    assert [tool.name for tool in get_non_retrieval_tools(settings)] == [
        CHECK_CONTRAST,
        CSS_SPECIFICITY,
        BROWSER_SUPPORT,
    ]
    for tool in tools[1:]:
        assert tool.description.strip()
        assert tool.tool_call_schema.model_json_schema()["properties"]


def test_building_the_tools_reads_no_data(settings: Settings) -> None:
    # The browser data does not exist under tmp_path: building works, the first call fails.
    browser_tool = get_non_retrieval_tools(settings)[2]

    with pytest.raises(ToolException, match="ingest --download"):
        browser_tool.invoke({"feature": ":has()"})
