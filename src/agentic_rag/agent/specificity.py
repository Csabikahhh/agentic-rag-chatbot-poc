"""CSS selector specificity by the rules of Selectors Level 4.

The logic behind the ``css_specificity`` tool (plan decision 9). The specificity of a complex
selector is the triple ``(a, b, c)``:

- ``a``: ID selectors (``#nav``);
- ``b``: class selectors (``.item``), attribute selectors (``[type="text"]``) and
  pseudo-classes (``:hover``);
- ``c``: type selectors (``a``, ``svg|circle``) and pseudo-elements (``::before``, and the
  legacy single-colon ``:before``, ``:after``, ``:first-line``, ``:first-letter``).

The universal selector ``*`` and the combinators add nothing. Some pseudo-classes take the
specificity of their argument instead of their own:

- ``:is()``, ``:not()`` and ``:has()``: the most specific complex selector of the argument;
- ``:where()``: zero;
- ``:nth-child(An+B of S)`` and ``:nth-last-child(An+B of S)``: one pseudo-class plus the
  most specific selector of ``S``;
- ``:host(S)``, ``:host-context(S)``: one pseudo-class plus ``S``; ``::slotted(S)``: one
  pseudo-element plus ``S``.

Triples compare lexicographically, and between equal specificities the declaration that
comes later in the stylesheet wins. Specificity decides only between declarations of the same
origin, importance and cascade layer; ``!important``, layers and inline ``style`` attributes
come first. A selector list (``a, b``) is not one selector: each complex selector has its own
specificity. The nesting selector ``&`` takes the specificity of its parent rule, which a
selector alone does not show, so it is rejected.

The module is pure Python: no LangChain, no I/O.
"""

import re
from dataclasses import dataclass
from typing import Final

__all__ = ["Specificity", "compare_specificity", "specificity", "split_selector_list"]

_LEGACY_PSEUDO_ELEMENTS: Final = frozenset({"before", "after", "first-line", "first-letter"})
_ARGUMENT_PSEUDO_CLASSES: Final = frozenset({"is", "not", "has", "matches", "-webkit-any"})
_NTH_OF: Final = frozenset({"nth-child", "nth-last-child"})
_IDENT: Final = re.compile(r"-?(?:[_a-zA-Z -￿]|\\.)(?:[-_a-zA-Z0-9 -￿]|\\.)*")
_COMBINATOR_CHARACTERS: Final = frozenset(" \t\n\r\f>+~")


@dataclass(frozen=True, order=True)
class Specificity:
    """The specificity triple; instances compare lexicographically."""

    ids: int = 0
    classes: int = 0
    types: int = 0

    def __add__(self, other: "Specificity") -> "Specificity":
        """Add two specificities component by component."""
        return Specificity(
            self.ids + other.ids, self.classes + other.classes, self.types + other.types
        )

    def __str__(self) -> str:
        """``(a, b, c)``."""
        return f"({self.ids}, {self.classes}, {self.types})"


_ID: Final = Specificity(1, 0, 0)
_CLASS: Final = Specificity(0, 1, 0)
_TYPE: Final = Specificity(0, 0, 1)
_ZERO: Final = Specificity()


def split_selector_list(selector_list: str) -> list[str]:
    """Split a selector list at its top-level commas.

    Args:
        selector_list: Selectors separated by commas, as in a rule's prelude.

    Returns:
        The complex selectors, stripped, without empty ones.
    """
    parts: list[str] = []
    depth = 0
    quote: str | None = None
    current: list[str] = []
    index = 0
    while index < len(selector_list):
        char = selector_list[index]
        if char == "\\" and index + 1 < len(selector_list):
            current.append(selector_list[index : index + 2])
            index += 2
            continue
        if quote:
            if char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append("".join(current))
            current = []
            index += 1
            continue
        current.append(char)
        index += 1
    parts.append("".join(current))
    return [part.strip() for part in parts if part.strip()]


def specificity(selector: str) -> Specificity:
    """Compute the specificity of one complex selector.

    Args:
        selector: A complex selector such as ``#nav > .item a:hover::after``.

    Returns:
        Its specificity.

    Raises:
        ValueError: If the text is empty, is a selector list, uses the nesting selector
            ``&``, or cannot be parsed.
    """
    text = selector.strip()
    if not text:
        raise ValueError("the selector is empty")
    if len(split_selector_list(text)) > 1:
        msg = f"{selector!r} is a selector list; give each selector on its own"
        raise ValueError(msg)
    return _Parser(text).parse()


def compare_specificity(selectors: list[str]) -> list[tuple[str, Specificity]]:
    """Compute the specificity of several selectors, in the order given.

    Selector lists are split into their complex selectors, which are listed one by one.

    Args:
        selectors: Selectors to compare.

    Returns:
        ``(selector, specificity)`` pairs in input order.

    Raises:
        ValueError: If no selector is given or one cannot be parsed.
    """
    pairs = [
        (part, specificity(part))
        for selector in selectors
        for part in split_selector_list(selector)
    ]
    if not pairs:
        raise ValueError("give at least one selector")
    return pairs


class _Parser:
    """A small recursive parser for the specificity of one complex selector."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.position = 0

    def parse(self) -> Specificity:
        """Walk the selector and sum the specificity of its simple selectors."""
        total = _ZERO
        while self.position < len(self.text):
            char = self.text[self.position]
            if char in _COMBINATOR_CHARACTERS or self.text.startswith("||", self.position):
                self.position += 2 if char == "|" else 1
            elif char == "#":
                self.position += 1
                self._ident("an ID")
                total += _ID
            elif char == ".":
                self.position += 1
                self._ident("a class name")
                total += _CLASS
            elif char == "[":
                self._skip_brackets()
                total += _CLASS
            elif char == ":":
                total += self._pseudo()
            elif char == "*":
                self.position += 1
                self._namespace_suffix(universal=True)
            elif char == "&":
                msg = "the nesting selector & takes the specificity of its parent rule"
                raise ValueError(msg)
            elif char == "|" or _IDENT.match(self.text, self.position):
                total += self._type_selector()
            else:
                msg = f"unexpected {char!r} at position {self.position} of {self.text!r}"
                raise ValueError(msg)
        return total

    def _ident(self, what: str) -> str:
        """Read an identifier at the current position."""
        match = _IDENT.match(self.text, self.position)
        if match is None:
            msg = f"expected {what} at position {self.position} of {self.text!r}"
            raise ValueError(msg)
        self.position = match.end()
        return match[0]

    def _type_selector(self) -> Specificity:
        """A type selector, possibly with a namespace prefix: ``a``, ``svg|a``, ``|a``."""
        if self.text[self.position] != "|":
            self._ident("a type selector")
        if self.text.startswith("|", self.position):
            self.position += 1
            if self.text.startswith("*", self.position):
                self.position += 1
                return _ZERO
            self._ident("a type selector")
        return _TYPE

    def _namespace_suffix(self, *, universal: bool) -> None:
        """Skip ``|x`` after ``*``: ``*|*`` adds nothing, ``*|a`` is handled as a type."""
        if self.text.startswith("|", self.position) and not self.text.startswith(
            "||", self.position
        ):
            self.position += 1
            if self.text.startswith("*", self.position):
                self.position += 1
            elif universal:
                self.position -= 1  # leave "|a" to _type_selector

    def _skip_brackets(self) -> None:
        """Skip an attribute selector, honouring quotes and escapes."""
        end = self._matching(self.position, "[", "]")
        self.position = end + 1

    def _pseudo(self) -> Specificity:
        """A pseudo-class or pseudo-element, with its argument if it has one."""
        element = self.text.startswith("::", self.position)
        self.position += 2 if element else 1
        name = self._ident("a pseudo-class or pseudo-element name").lower()
        argument: str | None = None
        if self.text.startswith("(", self.position):
            end = self._matching(self.position, "(", ")")
            argument = self.text[self.position + 1 : end]
            self.position = end + 1
        if element or name in _LEGACY_PSEUDO_ELEMENTS:
            return _TYPE + (self._max_of(argument) if name == "slotted" and argument else _ZERO)
        if name == "where":
            return _ZERO
        if name in _ARGUMENT_PSEUDO_CLASSES:
            return self._max_of(argument or "")
        if name in _NTH_OF and argument and re.search(r"\sof\s", argument):
            return _CLASS + self._max_of(re.split(r"\sof\s", argument, maxsplit=1)[1])
        if name in {"host", "host-context"} and argument:
            return _CLASS + self._max_of(argument)
        return _CLASS

    def _max_of(self, selector_list: str) -> Specificity:
        """The most specific complex selector of an argument list (relative ones included)."""
        parts = split_selector_list(selector_list)
        if not parts:
            msg = f"empty argument in {self.text!r}"
            raise ValueError(msg)
        return max(_Parser(part.lstrip(">+~ ")).parse() for part in parts)

    def _matching(self, start: int, opening: str, closing: str) -> int:
        """The index of the bracket that closes the one at ``start``."""
        depth = 0
        quote: str | None = None
        index = start
        while index < len(self.text):
            char = self.text[index]
            if char == "\\":
                index += 2
                continue
            if quote:
                if char == quote:
                    quote = None
            elif char in "\"'":
                quote = char
            elif char == opening:
                depth += 1
            elif char == closing:
                depth -= 1
                if depth == 0:
                    return index
            index += 1
        msg = f"unclosed {opening!r} in {self.text!r}"
        raise ValueError(msg)
