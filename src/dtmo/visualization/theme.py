"""One place for every colour and layout token the charts use.

The categorical slots are a validated palette: run through the colourblind and
contrast checks in both modes before adoption. Worst adjacent CVD separation is
dE 9.1 light / 8.4 dark against a target of 8, and worst normal-vision
separation dE 19.6 / 19.3 against a floor of 15.

Three light-mode slots sit below 3:1 contrast on the light surface, so charts
using them owe the reader *relief*: visible labels, hover tooltips, and the
table view the dashboard ships underneath the Gantt. Do not add a sixth family
colour by inventing a hue -- a generated one is indistinguishable from an
existing slot under colour-vision deficiency.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Product families, in a fixed order. Colour follows the family, never its
#: rank in a chart -- filtering the chart must not repaint the survivors.
FAMILY_ORDER: tuple[str, ...] = (
    "Wing Spar",
    "Fuselage Panel",
    "Landing Gear",
    "Avionics Bay",
    "Engine Mount",
)


@dataclass(frozen=True)
class Theme:
    """Colours for one render mode."""

    name: str
    surface: str
    surface_sunken: str
    text_primary: str
    text_secondary: str
    text_muted: str
    grid: str
    axis: str
    #: Categorical slots 1-5, assigned to families in FAMILY_ORDER.
    categorical: tuple[str, ...]
    #: Single hue for magnitude comparisons.
    sequential: str
    #: The de-emphasis grey for everything that is context, not the point.
    muted_mark: str
    #: Reserved for bounds and targets. Never used as a series colour.
    reference: str

    def family_colour(self, family: str) -> str:
        try:
            return self.categorical[FAMILY_ORDER.index(family)]
        except ValueError:
            return self.muted_mark


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    surface_sunken="#f2f2f0",
    text_primary="#0b0b0b",
    text_secondary="#52514e",
    text_muted="#78776f",
    grid="#e6e6e2",
    axis="#c9c9c3",
    categorical=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"),
    sequential="#2a78d6",
    muted_mark="#b9b9b2",
    reference="#0b0b0b",
)

DARK = Theme(
    name="dark",
    surface="#1a1a19",
    surface_sunken="#232322",
    text_primary="#ffffff",
    text_secondary="#c3c2b7",
    text_muted="#94938a",
    grid="#2f2f2d",
    axis="#4a4a46",
    categorical=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"),
    sequential="#3987e5",
    muted_mark="#5c5c57",
    reference="#ffffff",
)

THEMES = {"light": LIGHT, "dark": DARK}

FONT_STACK = "IBM Plex Sans, Segoe UI, system-ui, sans-serif"
MONO_STACK = "IBM Plex Mono, Cascadia Mono, Consolas, monospace"

#: Marks are thin and grids recessive; the data should be the loudest thing.
LINE_WIDTH = 2
MARKER_SIZE = 8
#: A surface-coloured gap between adjacent fills keeps bars from fusing.
BAR_GAP_PX = 2


def base_layout(theme: Theme, title: str, height: int = 420) -> dict:
    """Shared Plotly layout. Text always wears text tokens, never a series hue."""
    return {
        "title": {
            "text": title,
            "font": {"size": 15, "color": theme.text_primary, "family": FONT_STACK},
            "x": 0,
            "xanchor": "left",
            "pad": {"b": 12},
        },
        "paper_bgcolor": theme.surface,
        "plot_bgcolor": theme.surface,
        "font": {"family": FONT_STACK, "size": 12, "color": theme.text_secondary},
        "height": height,
        "margin": {"l": 8, "r": 20, "t": 52, "b": 44},
        "hoverlabel": {
            "bgcolor": theme.surface,
            "bordercolor": theme.axis,
            "font": {"family": MONO_STACK, "size": 12, "color": theme.text_primary},
        },
        "xaxis": {
            "gridcolor": theme.grid,
            "linecolor": theme.axis,
            "zeroline": False,
            "tickfont": {"color": theme.text_muted, "size": 11},
        },
        "yaxis": {
            "gridcolor": theme.grid,
            "linecolor": theme.axis,
            "zeroline": False,
            "tickfont": {"color": theme.text_muted, "size": 11},
        },
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "font": {"color": theme.text_secondary, "size": 11},
            "bgcolor": "rgba(0,0,0,0)",
        },
    }
