"""Plotly figures for the factory twin.

Each figure's form is chosen by the job the data has to do, not by habit:

* the Gantt tells *identity* apart (which family is on which machine), so it is
  the one chart that earns a categorical palette;
* the policy comparison has one point to make -- three policies are tied and the
  rest are far behind -- so it uses **emphasis**: the contenders in the accent
  hue, everything else in the de-emphasis grey, with the lower bound as a
  reference rule rather than a competing series;
* station load is pure magnitude, so it is a single-hue bar chart;
* the learning curve is one series over time, so it needs no legend at all --
  the title names it.

No chart here has two y-axes. Where two measures of different scale matter,
they get two charts.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import plotly.graph_objects as go

from ..digital_twin.entities import Job
from .theme import (
    BAR_GAP_PX,
    FAMILY_ORDER,
    LINE_WIDTH,
    MARKER_SIZE,
    MONO_STACK,
    Theme,
    base_layout,
)


def gantt(
    jobs: Sequence[Job],
    station_order: Sequence[str],
    theme: Theme,
    max_jobs: int | None = 60,
    title: str = "Schedule — who ran where, and when",
) -> go.Figure:
    """Operations as bars on station rows, coloured by product family.

    Shows the machines, not the jobs: each row is a work centre, so gaps in a
    row are idle capacity and dense stretches are contention. That is the thing
    worth seeing, and it is invisible in a KPI table.
    """
    shown = jobs if max_jobs is None else jobs[:max_jobs]

    figure = go.Figure()
    for family in FAMILY_ORDER:
        starts, durations, rows, labels = [], [], [], []
        for job in shown:
            if job.family.name != family:
                continue
            for station, start, finish in job.op_log:
                starts.append(start)
                durations.append(finish - start)
                rows.append(station)
                labels.append(
                    f"job {job.job_id} · {family}<br>"
                    f"{station}<br>"
                    f"{start:.1f}h → {finish:.1f}h ({finish - start:.1f}h)<br>"
                    f"due {job.due_date:.1f}h"
                )
        if not starts:
            continue
        figure.add_bar(
            x=durations,
            y=rows,
            base=starts,
            orientation="h",
            name=family,
            marker={
                "color": theme.family_colour(family),
                # A surface-coloured hairline keeps back-to-back operations from
                # reading as one long bar.
                "line": {"color": theme.surface, "width": BAR_GAP_PX},
            },
            hovertemplate="%{customdata}<extra></extra>",
            customdata=labels,
        )

    layout = base_layout(theme, title, height=380)
    layout["barmode"] = "overlay"
    layout["bargap"] = 0.35
    layout["xaxis"].update({"title": {"text": "hours", "font": {"size": 11}}})
    layout["yaxis"].update(
        {
            "categoryorder": "array",
            "categoryarray": list(reversed(station_order)),
            "gridcolor": "rgba(0,0,0,0)",
        }
    )
    layout["margin"]["l"] = 130
    figure.update_layout(**layout)
    return figure


def policy_comparison(
    means: Mapping[str, float],
    theme: Theme,
    highlight: Sequence[str] = (),
    lower_bound: float | None = None,
    title: str = "Weighted tardiness by policy",
    xaxis_title: str = "weighted tardiness (lower is better)",
) -> go.Figure:
    """Emphasis bar chart: the contenders in colour, the rest as context."""
    ordered = sorted(means.items(), key=lambda kv: kv[1], reverse=True)
    names = [name for name, _ in ordered]
    values = [value for _, value in ordered]
    highlight = set(highlight)
    colours = [
        theme.sequential if name in highlight else theme.muted_mark for name in names
    ]

    figure = go.Figure()
    figure.add_bar(
        x=values,
        y=names,
        orientation="h",
        marker={"color": colours, "line": {"color": theme.surface, "width": BAR_GAP_PX}},
        text=[f"{value:,.0f}" for value in values],
        textposition="outside",
        textfont={"family": MONO_STACK, "size": 11, "color": theme.text_secondary},
        hovertemplate="%{y}: %{x:,.1f}<extra></extra>",
    )

    layout = base_layout(theme, title, height=340)
    layout["showlegend"] = False
    layout["xaxis"].update(
        {"title": {"text": xaxis_title, "font": {"size": 11}}, "rangemode": "tozero"}
    )
    layout["yaxis"].update({"gridcolor": "rgba(0,0,0,0)"})
    layout["margin"]["l"] = 92
    layout["margin"]["r"] = 72

    if lower_bound is not None:
        # A bound is not a competing series, so it is drawn as a rule.
        figure.add_vline(
            x=lower_bound,
            line={"color": theme.reference, "width": 1, "dash": "dot"},
            annotation={
                "text": f"lower bound {lower_bound:,.0f}",
                "font": {"size": 10, "color": theme.text_muted},
                "yanchor": "bottom",
            },
        )
    figure.update_layout(**layout)
    return figure


def station_load(
    utilisation: Mapping[str, float],
    theme: Theme,
    expected: Mapping[str, float] | None = None,
    title: str = "Station utilisation",
) -> go.Figure:
    """Single-hue magnitude bars, with the analytic load as a marker."""
    names = list(utilisation)
    values = [utilisation[name] for name in names]

    figure = go.Figure()
    figure.add_bar(
        x=values,
        y=names,
        orientation="h",
        name="simulated",
        marker={
            "color": theme.sequential,
            "line": {"color": theme.surface, "width": BAR_GAP_PX},
        },
        text=[f"{value:.0%}" for value in values],
        textposition="outside",
        textfont={"family": MONO_STACK, "size": 11, "color": theme.text_secondary},
        hovertemplate="%{y}: %{x:.1%}<extra></extra>",
    )
    if expected:
        figure.add_scatter(
            x=[expected[name] for name in names],
            y=names,
            mode="markers",
            name="predicted before simulating",
            marker={
                "color": theme.text_primary,
                "size": MARKER_SIZE,
                "symbol": "line-ns",
                "line": {"color": theme.text_primary, "width": 2},
            },
            hovertemplate="%{y} predicted: %{x:.1%}<extra></extra>",
        )

    layout = base_layout(theme, title, height=320)
    layout["xaxis"].update(
        {
            "title": {"text": "share of machine-hours busy", "font": {"size": 11}},
            "tickformat": ".0%",
            "rangemode": "tozero",
        }
    )
    layout["yaxis"].update({"gridcolor": "rgba(0,0,0,0)", "autorange": "reversed"})
    layout["margin"]["l"] = 130
    layout["margin"]["r"] = 60
    layout["showlegend"] = bool(expected)
    figure.update_layout(**layout)
    return figure


def learning_curve(
    history: Sequence[Mapping[str, float]],
    theme: Theme,
    baseline_name: str = "spt",
    title: str = "PPO improvement over the baseline, during training",
) -> go.Figure:
    """One series over time. No legend -- the title says what it is."""
    steps = [row["timesteps"] for row in history]
    improvement = [row["improvement"] for row in history]

    figure = go.Figure()
    figure.add_scatter(
        x=steps,
        y=improvement,
        mode="lines+markers",
        line={"color": theme.sequential, "width": LINE_WIDTH},
        marker={
            "size": MARKER_SIZE,
            "color": theme.sequential,
            "line": {"color": theme.surface, "width": 2},
        },
        hovertemplate="%{x:,} steps<br>%{y:+.2f} vs " + baseline_name + "<extra></extra>",
    )

    layout = base_layout(theme, title, height=320)
    layout["showlegend"] = False
    layout["xaxis"].update({"title": {"text": "training timesteps", "font": {"size": 11}}})
    layout["yaxis"].update(
        {"title": {"text": f"paired return vs {baseline_name}", "font": {"size": 11}}}
    )
    layout["margin"]["l"] = 74
    layout["hovermode"] = "x unified"
    figure.update_layout(**layout)

    # Parity is the line that matters: above it the agent is winning.
    figure.add_hline(
        y=0,
        line={"color": theme.reference, "width": 1, "dash": "dot"},
        annotation={
            "text": f"parity with {baseline_name}",
            "font": {"size": 10, "color": theme.text_muted},
            "yanchor": "bottom",
        },
    )
    return figure
