"""WCAG 2.x colour contrast: parse CSS colours, compute the ratio and the conformance verdicts.

The logic behind the ``check_contrast`` tool (plan decision 9). It follows WCAG 2.2:

- The relative luminance of an sRGB colour is ``0.2126 R + 0.7152 G + 0.0722 B`` over the
  linearized channels (a channel ``c`` in 0..1 becomes ``c / 12.92`` up to 0.04045 and
  ``((c + 0.055) / 1.055) ** 2.4`` above).
- The contrast ratio of two colours is ``(L1 + 0.05) / (L2 + 0.05)``, the lighter one first;
  it ranges from 1:1 to 21:1.
- A semi-transparent foreground is composited over the background first, as the browser
  paints it; the background itself must be opaque.
- The ratio is reported rounded down to two decimals, so a ratio just below a threshold
  (4.499) is never shown as passing it (4.50).

Thresholds: success criterion 1.4.3 (AA) asks for 4.5:1 for normal text and 3:1 for large
text (at least 24 px, or 18.66 px bold); 1.4.6 (AAA) for 7:1 and 4.5:1; 1.4.11 (AA) for 3:1
for user-interface components and graphical objects.

Accepted colours: hex (``#rgb``, ``#rgba``, ``#rrggbb``, ``#rrggbbaa``), ``rgb()`` and
``rgba()`` with commas or spaces, percentages and a ``/`` alpha, ``hsl()`` and ``hsla()``, and
the CSS named colours of :data:`NAMED_COLORS`. Anything else raises ``ValueError``.

The module is pure Python: no LangChain, no I/O.
"""

import colorsys
import math
import re
from dataclasses import dataclass
from typing import Final

__all__ = [
    "NAMED_COLORS",
    "THRESHOLDS",
    "Color",
    "ContrastResult",
    "contrast_ratio",
    "evaluate_contrast",
    "parse_color",
    "relative_luminance",
]

NAMED_COLORS: Final[dict[str, str]] = {
    # The 16 basic colours of CSS, their aliases and common extended names.
    "black": "#000000",
    "silver": "#c0c0c0",
    "gray": "#808080",
    "grey": "#808080",
    "white": "#ffffff",
    "maroon": "#800000",
    "red": "#ff0000",
    "purple": "#800080",
    "fuchsia": "#ff00ff",
    "magenta": "#ff00ff",
    "green": "#008000",
    "lime": "#00ff00",
    "olive": "#808000",
    "yellow": "#ffff00",
    "navy": "#000080",
    "blue": "#0000ff",
    "teal": "#008080",
    "aqua": "#00ffff",
    "cyan": "#00ffff",
    "orange": "#ffa500",
    "darkgray": "#a9a9a9",
    "darkgrey": "#a9a9a9",
    "lightgray": "#d3d3d3",
    "lightgrey": "#d3d3d3",
    "dimgray": "#696969",
    "dimgrey": "#696969",
    "gainsboro": "#dcdcdc",
    "whitesmoke": "#f5f5f5",
    "darkblue": "#00008b",
    "darkred": "#8b0000",
    "darkgreen": "#006400",
    "rebeccapurple": "#663399",
}
"""CSS named colours the tool accepts, in lower case."""

THRESHOLDS: Final[tuple[tuple[str, str, float], ...]] = (
    ("AA", "normal text", 4.5),
    ("AA", "large text", 3.0),
    ("AAA", "normal text", 7.0),
    ("AAA", "large text", 4.5),
    ("AA", "UI components and graphics", 3.0),
)
"""(level, what, minimum ratio) of WCAG 2.2 success criteria 1.4.3, 1.4.6 and 1.4.11."""

_HEX: Final = re.compile(r"#([0-9a-f]{3,4}|[0-9a-f]{6}|[0-9a-f]{8})")
_FUNCTION: Final = re.compile(r"(rgba?|hsla?)\((.*)\)")
_NUMBER: Final = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?(%|deg|turn|rad|grad)?")


@dataclass(frozen=True)
class Color:
    """An sRGB colour: channels in 0..255 and alpha in 0..1."""

    red: float
    green: float
    blue: float
    alpha: float = 1.0

    @property
    def hex(self) -> str:
        """The colour as ``#rrggbb`` (alpha left out)."""
        return "#" + "".join(f"{round(channel):02x}" for channel in self.rgb)

    @property
    def rgb(self) -> tuple[float, float, float]:
        """The three channels."""
        return (self.red, self.green, self.blue)


@dataclass(frozen=True)
class ContrastResult:
    """The contrast of a foreground over a background.

    Attributes:
        foreground: The foreground as given, composited over the background when it was
            semi-transparent.
        background: The background.
        ratio: The contrast ratio, rounded down to two decimals.
        verdicts: ``(level, what, minimum, passes)`` for every row of :data:`THRESHOLDS`.
    """

    foreground: Color
    background: Color
    ratio: float
    verdicts: tuple[tuple[str, str, float, bool], ...]


def parse_color(text: str) -> Color:
    """Parse a CSS colour.

    Args:
        text: A colour in one of the formats of the module docstring.

    Returns:
        The colour.

    Raises:
        ValueError: If the format is not supported or a value is out of range.
    """
    value = text.strip().lower()
    if value in NAMED_COLORS:
        value = NAMED_COLORS[value]
    hex_match = _HEX.fullmatch(value)
    if hex_match:
        digits = hex_match[1]
        if len(digits) <= 4:
            digits = "".join(digit * 2 for digit in digits)
        channels = [int(digits[i : i + 2], 16) for i in range(0, len(digits), 2)]
        alpha = channels[3] / 255 if len(channels) == 4 else 1.0
        return Color(channels[0], channels[1], channels[2], alpha)
    function = _FUNCTION.fullmatch(value)
    if function:
        return _parse_function(function[1], function[2], text)
    msg = (
        f"unsupported colour {text!r}: use hex (#777 or #777777), rgb(), rgba(), hsl(), "
        f"hsla() or a basic CSS colour name such as white or black"
    )
    raise ValueError(msg)


def relative_luminance(color: Color) -> float:
    """The WCAG relative luminance of an opaque colour, from 0 (black) to 1 (white)."""
    red, green, blue = (_linear(channel / 255) for channel in color.rgb)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast_ratio(foreground: Color, background: Color) -> float:
    """The exact WCAG contrast ratio of a foreground painted over an opaque background.

    Raises:
        ValueError: If the background is not opaque.
    """
    if background.alpha < 1:
        msg = "the background must be opaque; give the colour it is painted on"
        raise ValueError(msg)
    painted = _composite(foreground, background)
    lighter, darker = sorted(
        (relative_luminance(painted), relative_luminance(background)), reverse=True
    )
    return (lighter + 0.05) / (darker + 0.05)


def evaluate_contrast(foreground: str, background: str) -> ContrastResult:
    """Parse two colours and evaluate their contrast against every WCAG threshold.

    Args:
        foreground: The text or icon colour.
        background: The colour behind it.

    Returns:
        The result, with the ratio rounded down to two decimals and the verdicts computed from
        the exact ratio.

    Raises:
        ValueError: If a colour cannot be parsed or the background is not opaque.
    """
    fg, bg = parse_color(foreground), parse_color(background)
    exact = contrast_ratio(fg, bg)
    verdicts = tuple(
        (level, what, minimum, exact >= minimum) for level, what, minimum in THRESHOLDS
    )
    return ContrastResult(
        foreground=_composite(fg, bg),
        background=bg,
        ratio=math.floor(exact * 100) / 100,
        verdicts=verdicts,
    )


def _linear(channel: float) -> float:
    """Linearize one sRGB channel in 0..1."""
    return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4


def _composite(foreground: Color, background: Color) -> Color:
    """Paint a possibly semi-transparent foreground over an opaque background."""
    if foreground.alpha >= 1:
        return foreground
    alpha = foreground.alpha
    channels = [
        alpha * front + (1 - alpha) * back
        for front, back in zip(foreground.rgb, background.rgb, strict=True)
    ]
    return Color(channels[0], channels[1], channels[2])


def _parse_function(name: str, body: str, text: str) -> Color:
    """Parse the arguments of ``rgb()``, ``rgba()``, ``hsl()`` or ``hsla()``."""
    parts = [part for part in re.split(r"[\s,/]+", body.strip()) if part]
    if len(parts) not in (3, 4) or not all(_NUMBER.fullmatch(part) for part in parts):
        msg = f"cannot read the colour {text!r}"
        raise ValueError(msg)
    alpha = _alpha(parts[3]) if len(parts) == 4 else 1.0
    if name.startswith("rgb"):
        channels = [_channel(part) for part in parts[:3]]
        return Color(channels[0], channels[1], channels[2], alpha)
    hue = _hue(parts[0])
    saturation, lightness = (_fraction(part) for part in parts[1:3])
    red, green, blue = colorsys.hls_to_rgb(hue, lightness, saturation)
    return Color(red * 255, green * 255, blue * 255, alpha)


def _channel(part: str) -> float:
    """An rgb() channel: 0..255, or a percentage."""
    value = float(part.rstrip("%"))
    value = value * 2.55 if part.endswith("%") else value
    if not 0 <= value <= 255:
        msg = f"colour channel out of range: {part}"
        raise ValueError(msg)
    return value


def _alpha(part: str) -> float:
    """An alpha value: 0..1, or a percentage."""
    value = float(part.rstrip("%")) / (100 if part.endswith("%") else 1)
    if not 0 <= value <= 1:
        msg = f"alpha out of range: {part}"
        raise ValueError(msg)
    return value


def _fraction(part: str) -> float:
    """A saturation or lightness: a percentage (or a bare number, as CSS Color 4 allows)."""
    value = float(part.rstrip("%")) / 100
    if not 0 <= value <= 1:
        msg = f"saturation or lightness out of range: {part}"
        raise ValueError(msg)
    return value


def _hue(part: str) -> float:
    """A hue as a fraction of a turn."""
    number = float(re.sub(r"(deg|turn|rad|grad)$", "", part))
    if part.endswith("turn"):
        degrees = number * 360
    elif part.endswith("grad"):
        degrees = number * 0.9
    elif part.endswith("rad"):
        degrees = math.degrees(number)
    else:
        degrees = number
    return (degrees % 360) / 360
