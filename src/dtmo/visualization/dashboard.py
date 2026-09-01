"""Assemble the figures into one self-contained HTML dashboard.

Dark mode is *selected*, not flipped: both theme's figures are rendered from
their own validated palettes and the page shows one. A naive inversion would
push the categorical hues out of the lightness band they were checked in.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from typing import Mapping, Sequence

import plotly.graph_objects as go
import plotly.io as pio

from .theme import DARK, FAMILY_ORDER, LIGHT, MONO_STACK, Theme

PLOTLY_CDN = "https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.32.0/plotly.min.js"


@dataclass(frozen=True)
class StatTile:
    """A headline number. A single value is a tile, never a one-bar chart."""

    label: str
    value: str
    note: str = ""


def _figure_div(figure: go.Figure, div_id: str) -> str:
    return pio.to_html(
        figure,
        include_plotlyjs=False,
        full_html=False,
        div_id=div_id,
        config={"displayModeBar": False, "responsive": True},
    )


def _tiles_html(tiles: Sequence[StatTile]) -> str:
    cells = "".join(
        f'<div class="tile"><span class="tile-k">{html.escape(t.label)}</span>'
        f'<span class="tile-v">{html.escape(t.value)}</span>'
        f'<span class="tile-n">{html.escape(t.note)}</span></div>'
        for t in tiles
    )
    return f'<div class="tiles">{cells}</div>'


def _family_table(counts: Mapping[str, int], theme_pair: tuple[Theme, Theme]) -> str:
    """Identity in text, not colour alone.

    Three light-mode categorical slots sit under 3:1 contrast on the light
    surface, which obliges relief: this table is it.
    """
    light, dark = theme_pair
    rows = []
    for family in FAMILY_ORDER:
        rows.append(
            "<tr>"
            f'<td><span class="sw" style="--l:{light.family_colour(family)};'
            f'--d:{dark.family_colour(family)}"></span>{html.escape(family)}</td>'
            f'<td class="num">{counts.get(family, 0)}</td>'
            "</tr>"
        )
    return (
        '<table class="ftable"><thead><tr><th>Product family</th>'
        '<th class="num">Operations shown</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def build_dashboard(
    figures: Mapping[str, tuple[go.Figure, go.Figure]],
    tiles: Sequence[StatTile],
    family_counts: Mapping[str, int],
    title: str = "Factory Twin Dashboard",
    subtitle: str = "",
    notes: Mapping[str, str] | None = None,
) -> str:
    """Render every (light, dark) figure pair into one page.

    ``figures`` maps a slug to the pair; ``notes`` optionally maps the same slug
    to a sentence explaining what the reader should take from it.
    """
    notes = notes or {}
    blocks = []
    for slug, (light_fig, dark_fig) in figures.items():
        note = notes.get(slug, "")
        note_html = f'<p class="note">{html.escape(note)}</p>' if note else ""
        blocks.append(
            f'<section class="panel">'
            f'<div class="fig light-only">{_figure_div(light_fig, f"{slug}-light")}</div>'
            f'<div class="fig dark-only">{_figure_div(dark_fig, f"{slug}-dark")}</div>'
            f"{note_html}</section>"
        )
        if slug == "gantt":
            blocks.append(
                '<section class="panel">'
                + _family_table(family_counts, (LIGHT, DARK))
                + "</section>"
            )

    return _PAGE.format(
        title=html.escape(title),
        subtitle=html.escape(subtitle),
        plotly=PLOTLY_CDN,
        tiles=_tiles_html(tiles),
        body="\n".join(blocks),
        light_surface=LIGHT.surface,
        dark_surface=DARK.surface,
        ids=json.dumps(
            [f"{slug}-{mode}" for slug in figures for mode in ("light", "dark")]
        ),
    )


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<script src="{plotly}"></script>
<style>
:root {{
  --surface: {light_surface};
  --sunken: #f2f2f0;
  --ink: #0b0b0b;
  --ink-2: #52514e;
  --muted: #78776f;
  --line: #e6e6e2;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --surface: {dark_surface};
    --sunken: #232322;
    --ink: #ffffff;
    --ink-2: #c3c2b7;
    --muted: #94938a;
    --line: #2f2f2d;
  }}
}}
:root[data-theme="dark"] {{
  --surface: {dark_surface};
  --sunken: #232322;
  --ink: #ffffff;
  --ink-2: #c3c2b7;
  --muted: #94938a;
  --line: #2f2f2d;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; padding: 0 24px 80px;
  background: var(--surface); color: var(--ink);
  font-family: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
  font-size: 15px; line-height: 1.6;
}}
.wrap {{ max-width: 1080px; margin: 0 auto; }}
header {{ padding: 48px 0 22px; border-bottom: 2px solid var(--ink); }}
h1 {{ font-size: 30px; font-weight: 600; letter-spacing: -0.02em; margin: 0 0 6px; }}
.sub {{ color: var(--ink-2); margin: 0; max-width: 70ch; }}

.tiles {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 0; margin: 28px 0 8px; border: 1px solid var(--line);
}}
.tile {{ padding: 14px 18px; border-right: 1px solid var(--line); }}
.tile:last-child {{ border-right: none; }}
.tile-k {{
  display: block; font-family: {mono}; font-size: 10px; letter-spacing: 0.11em;
  text-transform: uppercase; color: var(--muted); margin-bottom: 5px;
}}
.tile-v {{
  display: block; font-size: 26px; font-weight: 600; letter-spacing: -0.02em;
  font-variant-numeric: tabular-nums;
}}
.tile-n {{ display: block; font-size: 12px; color: var(--muted); margin-top: 2px; }}

.panel {{ margin-top: 26px; border: 1px solid var(--line); background: var(--surface); }}
.fig {{ padding: 6px 6px 0; }}
.note {{
  margin: 0; padding: 12px 20px 16px; color: var(--ink-2);
  font-size: 13.5px; border-top: 1px solid var(--line); max-width: 78ch;
}}

.ftable {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
.ftable th {{
  text-align: left; padding: 11px 20px; border-bottom: 1px solid var(--line);
  font-family: {mono}; font-size: 10px; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--muted); font-weight: 500;
}}
.ftable td {{ padding: 9px 20px; border-bottom: 1px solid var(--line); }}
.ftable tr:last-child td {{ border-bottom: none; }}
.ftable .num {{ text-align: right; font-family: {mono}; font-variant-numeric: tabular-nums; }}
.sw {{
  display: inline-block; width: 11px; height: 11px; margin-right: 9px;
  background: var(--l); vertical-align: baseline;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) .sw {{ background: var(--d); }}
}}
:root[data-theme="dark"] .sw {{ background: var(--d); }}

.dark-only {{ display: none; }}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) .light-only {{ display: none; }}
  :root:not([data-theme="light"]) .dark-only {{ display: block; }}
}}
:root[data-theme="dark"] .light-only {{ display: none; }}
:root[data-theme="dark"] .dark-only {{ display: block; }}
:root[data-theme="light"] .light-only {{ display: block; }}
:root[data-theme="light"] .dark-only {{ display: none; }}

#themer {{
  position: fixed; top: 14px; right: 18px; z-index: 10;
  font-family: {mono}; font-size: 11px; letter-spacing: 0.06em;
  text-transform: uppercase; padding: 7px 12px; cursor: pointer;
  background: var(--sunken); color: var(--ink-2);
  border: 1px solid var(--line);
}}
#themer:focus-visible {{ outline: 2px solid var(--ink); outline-offset: 2px; }}
@media (prefers-reduced-motion: reduce) {{ * {{ animation: none !important; transition: none !important; }} }}
</style>
</head>
<body>
<button id="themer" type="button" aria-label="Switch colour theme">theme</button>
<div class="wrap">
  <header>
    <h1>{title}</h1>
    <p class="sub">{subtitle}</p>
  </header>
  {tiles}
  {body}
</div>
<script>
(function () {{
  var ids = {ids};
  function resize() {{
    ids.forEach(function (id) {{
      var el = document.getElementById(id);
      if (el && el.offsetParent !== null && window.Plotly) {{
        window.Plotly.Plots.resize(el);
      }}
    }});
  }}
  document.getElementById('themer').addEventListener('click', function () {{
    var root = document.documentElement;
    var isDark = root.getAttribute('data-theme') === 'dark'
      || (!root.getAttribute('data-theme')
          && window.matchMedia('(prefers-color-scheme: dark)').matches);
    root.setAttribute('data-theme', isDark ? 'light' : 'dark');
    setTimeout(resize, 0);
  }});
  window.addEventListener('resize', resize);
  setTimeout(resize, 0);
}})();
</script>
</body>
</html>
"""

_PAGE = _PAGE.replace("{mono}", MONO_STACK)
