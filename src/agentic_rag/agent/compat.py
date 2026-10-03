"""Browser support from MDN's browser-compat-data (BCD).

The logic behind the ``browser_support`` tool (plan decision 9). BCD is downloaded like the
corpus, at a pinned commit, into ``DATA_DIR/browser-compat-data/`` (``data/sources.toml``,
``index = false``, so it is never indexed). Its JSON files nest the features by area: the file
``css/selectors/has.json`` holds ``{"css": {"selectors": {"has": {"__compat": {...}}}}}``,
and a feature's dotted path, its *BCD id*, is ``css.selectors.has``. Each ``__compat`` holds a
``support`` statement per browser, the ``status`` flags and an ``mdn_url``; ``browsers/*.json``
describes the browsers and their releases.

:class:`CompatData` loads the files once, lazily, and indexes every feature by its id.
:meth:`CompatData.find` resolves a name as people write it: a BCD id (``css.selectors.has``),
a selector (``:has()``, ``::before``), an at-rule (``@container``), an element
(``<dialog>``), a dotted JavaScript or Web API name (``Promise.withResolvers``) or a bare
name (``grid-template-areas``, ``fetch``). An optional area (``css``, ``html``,
``javascript``, ``api``) narrows the search; among several matches the shortest id wins, and
the others are reported as alternatives.

Reading a support statement:

- A browser's statement is one entry or a list of entries; the primary entry is the first one
  without a flag, a vendor prefix, an alternative name or a partial implementation, otherwise
  the first entry.
- ``version_added`` is a version string (``"105"``, ``"15.4"``), a range (``"≤79"``: in 79 or
  earlier), ``"preview"``, ``true`` (supported, version unknown), ``false`` or ``null``
  (unknown).
- ``"mirror"`` instead of a statement means: as in the browser's ``upstream`` (Edge and
  Chrome Android follow Chrome, Safari on iOS follows Safari, ...). It is resolved the way
  BCD's own build resolves it: the upstream version becomes the first release of the browser
  with the same engine and at least the same engine version.
- A target browser version supports a feature when the primary entry was added in that
  version or earlier and not removed by it.

The module reads JSON files and does nothing else: no network, no LangChain.
"""

import json
import re
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

__all__ = [
    "AREAS",
    "BCD_DIRECTORY",
    "BROWSERS",
    "SUMMARY_BROWSERS",
    "CompatData",
    "Feature",
    "SupportVerdict",
    "parse_target",
]

BCD_DIRECTORY: Final = "browser-compat-data"
"""Directory of the downloaded BCD under ``DATA_DIR``; the source id in ``sources.toml``."""

type Area = Literal["css", "html", "javascript", "api"]
AREAS: Final[tuple[Area, ...]] = ("css", "html", "javascript", "api")
"""The BCD areas the tool searches."""

BROWSERS: Final[dict[str, str]] = {
    "chrome": "Chrome",
    "edge": "Edge",
    "firefox": "Firefox",
    "safari": "Safari",
    "chrome_android": "Chrome Android",
    "safari_ios": "Safari on iOS",
    "firefox_android": "Firefox for Android",
    "samsunginternet_android": "Samsung Internet",
    "opera": "Opera",
}
"""BCD browser keys the tool understands, with their display names."""

SUMMARY_BROWSERS: Final = ("chrome", "edge", "firefox", "safari", "chrome_android", "safari_ios")
"""The browsers of a support summary."""

_BROWSER_ALIASES: Final[dict[str, str]] = {
    "chrome": "chrome",
    "google chrome": "chrome",
    "edge": "edge",
    "microsoft edge": "edge",
    "firefox": "firefox",
    "safari": "safari",
    "opera": "opera",
    "chrome android": "chrome_android",
    "android chrome": "chrome_android",
    "chrome for android": "chrome_android",
    "safari ios": "safari_ios",
    "ios safari": "safari_ios",
    "safari on ios": "safari_ios",
    "ios": "safari_ios",
    "firefox android": "firefox_android",
    "firefox for android": "firefox_android",
    "samsung": "samsunginternet_android",
    "samsung internet": "samsunginternet_android",
}
_TARGET: Final = re.compile(r"(?P<name>[a-z][a-z ]*?)\s*(?:>=|≥|v)?\s*(?P<version>\d+(?:\.\d+)*)")
_VERSION: Final = re.compile(r"(?P<range>≤)?(?P<number>\d+(?:\.\d+)*)")
_NAME_DECORATION: Final = re.compile(r"^[<@:`'\"]+|[>()`'\"]+$")


@dataclass(frozen=True)
class Feature:
    """One BCD feature.

    Attributes:
        id: The dotted BCD id, for example ``css.selectors.has``.
        compat: Its ``__compat`` object.
    """

    id: str
    compat: Mapping[str, Any]


@dataclass(frozen=True)
class SupportVerdict:
    """Whether a target browser version supports a feature.

    Attributes:
        browser: The BCD browser key.
        version: The target version, as given.
        supported: True, False, or None when the data cannot tell.
        detail: A short explanation, for example ``added in 15.4``.
    """

    browser: str
    version: str
    supported: bool | None
    detail: str


@dataclass(frozen=True)
class _Release:
    """One release of a browser: its version and its rendering engine."""

    version: str
    engine: str
    engine_version: tuple[int, ...]


@dataclass(frozen=True)
class _Browser:
    """A browser of ``browsers/*.json``: the browser it mirrors and its releases."""

    upstream: str | None
    releases: dict[str, _Release]


def parse_target(text: str) -> tuple[str, str]:
    """Parse a target browser such as ``Safari 15``, ``iOS Safari 16.4`` or ``chrome>=110``.

    Args:
        text: A browser name and a version.

    Returns:
        ``(BCD browser key, version)``.

    Raises:
        ValueError: If the text has no version or names an unknown browser.
    """
    match = _TARGET.fullmatch(text.strip().lower())
    if match is None:
        msg = f"cannot read the target browser {text!r}: write a name and a version, e.g. Safari 15"
        raise ValueError(msg)
    name = " ".join(match["name"].replace("_", " ").split())
    if name not in _BROWSER_ALIASES:
        known = ", ".join(sorted({BROWSERS[key] for key in _BROWSER_ALIASES.values()}))
        msg = f"unknown browser {match['name'].strip()!r}; known browsers: {known}"
        raise ValueError(msg)
    return _BROWSER_ALIASES[name], match["version"]


class CompatData:
    """The BCD features and browsers of one download directory, loaded on first use.

    Loading parses the JSON files of :data:`AREAS` and of ``browsers/`` once (about 2 300
    files, well under a second) under a lock, so concurrent tool calls share one index.
    """

    def __init__(self, directory: Path) -> None:
        """Remember the download directory; nothing is read until the first lookup."""
        self.directory = directory
        self._loaded: tuple[dict[str, Feature], dict[str, _Browser]] | None = None
        self._lock = threading.Lock()

    @property
    def features(self) -> dict[str, Feature]:
        """Every feature by its BCD id.

        Raises:
            FileNotFoundError: If the data has not been downloaded.
        """
        return self._data()[0]

    def find(self, name: str, area: Area | None = None) -> list[Feature]:
        """Resolve a feature name to its BCD features, best match first.

        Args:
            name: A BCD id or a name as written in the documentation (see the module
                docstring).
            area: Search only this area.

        Returns:
            The matching features, best first; empty when nothing matches.
        """
        features = self.features
        text = name.strip().strip("`").strip()
        if text in features:
            return [features[text]]
        prefixes = _prefixes_for(text, area)
        key = _NAME_DECORATION.sub("", text).strip().lower()
        if not key:
            return []
        dotted = "." + key.replace(" ", "-")
        matches = [
            feature
            for feature_id, feature in features.items()
            if feature_id.lower().endswith(dotted)
            and (not prefixes or feature_id.startswith(prefixes))
        ]
        return sorted(matches, key=lambda feature: (feature.id.count("."), feature.id))

    def entries(self, feature: Feature, browser: str) -> list[Mapping[str, Any]]:
        """The support entries of a feature in a browser, with ``"mirror"`` resolved."""
        statement = feature.compat.get("support", {}).get(browser)
        if statement == "mirror":
            upstream = self._data()[1].get(browser, _Browser(None, {})).upstream
            if upstream is None:
                return []
            return [
                self._mirror(entry, upstream, browser) for entry in self.entries(feature, upstream)
            ]
        entries = statement if isinstance(statement, list) else [statement] if statement else []
        return [entry for entry in entries if isinstance(entry, Mapping)]

    def summary(self, feature: Feature) -> list[str]:
        """One line per summary browser: ``Safari: since 15.4``, ``Firefox: not supported``."""
        return [
            f"{BROWSERS[browser]}: {_describe(_primary(self.entries(feature, browser)))}"
            for browser in SUMMARY_BROWSERS
        ]

    def check(self, feature: Feature, browser: str, version: str) -> SupportVerdict:
        """Decide whether one target browser version supports a feature.

        Args:
            feature: The feature.
            browser: A BCD browser key from :func:`parse_target`.
            version: The target version.

        Returns:
            The verdict.
        """
        return _check_entry(_primary(self.entries(feature, browser)), browser, version)

    def _data(self) -> tuple[dict[str, Feature], dict[str, _Browser]]:
        """Load the data once."""
        if self._loaded is None:
            with self._lock:
                if self._loaded is None:
                    self._loaded = self._load()
        return self._loaded

    def _load(self) -> tuple[dict[str, Feature], dict[str, _Browser]]:
        """Parse the JSON files of every area and of the browsers."""
        if not self.directory.is_dir():
            msg = (
                f"The browser compatibility data is missing in {self.directory}; run "
                "'agentic-rag ingest --download'"
            )
            raise FileNotFoundError(msg)
        features: dict[str, Feature] = {}
        for area in AREAS:
            for path in sorted((self.directory / area).rglob("*.json")):
                data = json.loads(path.read_text(encoding="utf-8"))
                for feature_id, compat in _walk(data, ()):
                    features[feature_id] = Feature(id=feature_id, compat=compat)
        browsers: dict[str, _Browser] = {}
        for path in sorted((self.directory / "browsers").glob("*.json")):
            for key, info in json.loads(path.read_text(encoding="utf-8"))["browsers"].items():
                releases = {
                    version: _Release(
                        version, release.get("engine", ""), _numbers(release.get("engine_version"))
                    )
                    for version, release in info.get("releases", {}).items()
                }
                browsers[key] = _Browser(info.get("upstream"), releases)
        return features, browsers

    def _mirror(self, entry: Mapping[str, Any], upstream: str, browser: str) -> dict[str, Any]:
        """Translate an upstream entry into a browser's versions."""
        mirrored = dict(entry)
        for key in ("version_added", "version_removed"):
            if key in entry:
                mirrored[key] = self._mirror_version(entry[key], upstream, browser)
        return mirrored

    def _mirror_version(self, value: object, upstream: str, browser: str) -> object:
        """The browser's first release with the engine of an upstream release."""
        match = _VERSION.fullmatch(value) if isinstance(value, str) else None
        if match is None:
            return value
        browsers = self._data()[1]
        source = browsers.get(upstream, _Browser(None, {})).releases.get(match["number"])
        target = browsers.get(browser)
        if source is None or target is None or not source.engine_version:
            return value
        candidates = [
            release
            for release in target.releases.values()
            if release.engine == source.engine and release.engine_version >= source.engine_version
        ]
        if not candidates:
            return False
        first = min(candidates, key=lambda release: _version_tuple(release.version))
        return ("≤" if match["range"] else "") + first.version


def _check_entry(entry: Mapping[str, Any] | None, browser: str, version: str) -> SupportVerdict:
    """Decide support for one target version from the primary support entry."""
    if entry is None:
        return SupportVerdict(browser, version, None, "no data")
    added = entry.get("version_added")
    removed = entry.get("version_removed")
    if added is False:
        return SupportVerdict(browser, version, False, "not supported")
    if added is None:
        return SupportVerdict(browser, version, None, "support unknown")
    if added is True:
        return SupportVerdict(browser, version, True, "supported (version unknown)")
    if added == "preview":
        return SupportVerdict(browser, version, False, "only in preview builds")
    parsed = _VERSION.fullmatch(str(added))
    if parsed is None:
        return SupportVerdict(browser, version, None, f"version_added {added!r} not understood")
    target = _version_tuple(version)
    removed_match = _VERSION.fullmatch(str(removed)) if removed else None
    if removed_match and target >= _version_tuple(removed_match["number"]):
        return SupportVerdict(browser, version, False, f"removed in {removed}")
    notes = _notes(entry)
    if target >= _version_tuple(parsed["number"]):
        return SupportVerdict(browser, version, True, f"added in {added}{notes}")
    if parsed["range"]:
        return SupportVerdict(browser, version, None, f"added in {added}, exact version unknown")
    return SupportVerdict(browser, version, False, f"added in {added}{notes}")


def _walk(
    node: Mapping[str, Any], path: tuple[str, ...]
) -> Iterator[tuple[str, Mapping[str, Any]]]:
    """Yield ``(id, __compat)`` for every feature below a JSON node."""
    for key, value in node.items():
        if key == "__compat" and path:
            yield ".".join(path), value
        elif isinstance(value, Mapping) and not key.startswith("__"):
            yield from _walk(value, (*path, key))


def _prefixes_for(text: str, area: Area | None) -> tuple[str, ...]:
    """The id prefixes a name's form and the requested area allow."""
    if text.startswith(":"):
        return ("css.selectors.",)
    if text.startswith("@"):
        return ("css.at-rules.",)
    if text.startswith("<") and text.endswith(">"):
        return ("html.elements.",)
    return (f"{area}.",) if area else ()


def _primary(entries: list[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """The entry that describes plain, full support, else the first entry."""
    for entry in entries:
        if not any(
            entry.get(key)
            for key in ("flags", "prefix", "alternative_name", "partial_implementation")
        ):
            return entry
    return entries[0] if entries else None


def _describe(entry: Mapping[str, Any] | None) -> str:
    """A support entry in a few words."""
    if entry is None:
        return "no data"
    added = entry.get("version_added")
    if added is False:
        return "not supported"
    if added is None:
        return "unknown"
    if added is True:
        text = "supported (version unknown)"
    elif added == "preview":
        text = "preview builds only"
    else:
        text = f"since {added}"
    if entry.get("version_removed"):
        text += f", removed in {entry['version_removed']}"
    return text + _notes(entry)


def _notes(entry: Mapping[str, Any]) -> str:
    """Remarks about a support entry: partial support, a prefix, a flag."""
    remarks = []
    if entry.get("partial_implementation"):
        remarks.append("partial support")
    if entry.get("prefix"):
        remarks.append(f"with the {entry['prefix']} prefix")
    if entry.get("alternative_name"):
        remarks.append(f"as {entry['alternative_name']}")
    if entry.get("flags"):
        remarks.append("behind a flag")
    return f" ({', '.join(remarks)})" if remarks else ""


def _numbers(version: object) -> tuple[int, ...]:
    """The numeric parts of an engine version; empty when it has none."""
    return tuple(int(part) for part in re.findall(r"\d+", str(version))) if version else ()


def _version_tuple(version: str) -> tuple[int, ...]:
    """``"15.4"`` -> ``(15, 4)``; trailing zeros do not matter (``15`` equals ``15.0``)."""
    parts = [int(part) for part in re.findall(r"\d+", version)] or [0]
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)
