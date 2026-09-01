"""Charts and the dashboard.

These guard the rules that are easy to break silently in a refactor: colour
follows the entity rather than its rank, no chart grows a second y-axis, dark
mode is a selected palette rather than an inversion, and the light-mode
contrast shortfall keeps its relief.
"""

import json

import pytest

from dtmo.agents.policies import classical_policies
from dtmo.digital_twin.factory import FactoryModel
from dtmo.visualization.charts import (
    gantt,
    learning_curve,
    policy_comparison,
    station_load,
)
from dtmo.visualization.dashboard import StatTile, build_dashboard
from dtmo.visualization.theme import DARK, FAMILY_ORDER, LIGHT


@pytest.fixture(scope="module")
def run(config):
    model = FactoryModel(config.with_overrides(n_jobs=40), seed=1000)
    model.run()
    return model


@pytest.fixture
def tardiness():
    return {
        "spt": 2780.0,
        "blend": 2716.0,
        "lwkr": 3406.0,
        "lpt": 7554.0,
        "mwkr": 7568.0,
    }


@pytest.fixture
def history():
    return [
        {"timesteps": 25000, "improvement": -36.0, "win_rate": 0.06},
        {"timesteps": 50000, "improvement": -7.2, "win_rate": 0.25},
        {"timesteps": 75000, "improvement": 1.7, "win_rate": 0.5},
    ]


class TestTheme:
    def test_five_families_get_five_slots(self):
        assert len(FAMILY_ORDER) == 5
        assert len(LIGHT.categorical) >= 5
        assert len(DARK.categorical) >= 5

    def test_colour_follows_the_family_not_its_position(self):
        """A filtered chart must not repaint the survivors."""
        first = LIGHT.family_colour("Wing Spar")
        assert LIGHT.family_colour("Wing Spar") == first
        assert LIGHT.family_colour("Engine Mount") != first

    def test_slots_are_distinct_in_both_modes(self):
        for theme in (LIGHT, DARK):
            used = [theme.family_colour(f) for f in FAMILY_ORDER]
            assert len(set(used)) == len(used), theme.name

    def test_dark_is_a_separate_palette_not_an_inversion(self):
        light = [LIGHT.family_colour(f) for f in FAMILY_ORDER]
        dark = [DARK.family_colour(f) for f in FAMILY_ORDER]
        assert light != dark

    def test_unknown_family_falls_back_to_the_muted_mark(self):
        assert LIGHT.family_colour("Nonexistent") == LIGHT.muted_mark

    def test_reference_colour_is_not_a_series_colour(self):
        """Bounds and targets must not read as another series."""
        assert LIGHT.reference not in LIGHT.categorical
        assert DARK.reference not in DARK.categorical


class TestGantt:
    def test_one_trace_per_family_present(self, run):
        figure = gantt(run.completed, list(run.stations), LIGHT, max_jobs=40)
        names = {trace.name for trace in figure.data}
        assert names
        assert names.issubset(set(FAMILY_ORDER))

    def test_bars_are_anchored_at_their_start_time(self, run):
        figure = gantt(run.completed, list(run.stations), LIGHT, max_jobs=10)
        trace = figure.data[0]
        assert trace.base is not None
        assert len(trace.base) == len(trace.x)
        assert all(duration > 0 for duration in trace.x)

    def test_max_jobs_limits_what_is_drawn(self, run):
        few = gantt(run.completed, list(run.stations), LIGHT, max_jobs=5)
        many = gantt(run.completed, list(run.stations), LIGHT, max_jobs=40)
        assert sum(len(t.x) for t in few.data) < sum(len(t.x) for t in many.data)

    def test_every_bar_carries_a_hover_label(self, run):
        figure = gantt(run.completed, list(run.stations), LIGHT, max_jobs=10)
        for trace in figure.data:
            assert trace.customdata is not None
            assert len(trace.customdata) == len(trace.x)

    def test_bars_are_separated_by_a_surface_gap(self, run):
        figure = gantt(run.completed, list(run.stations), LIGHT, max_jobs=10)
        assert figure.data[0].marker.line.color == LIGHT.surface

    def test_station_rows_follow_the_configured_order(self, run):
        order = list(run.stations)
        figure = gantt(run.completed, order, LIGHT, max_jobs=10)
        assert list(figure.layout.yaxis.categoryarray) == list(reversed(order))


class TestPolicyComparison:
    def test_highlighted_policies_get_the_accent(self, tardiness):
        figure = policy_comparison(tardiness, LIGHT, highlight=["spt", "blend"])
        colours = dict(zip(figure.data[0].y, figure.data[0].marker.color))
        assert colours["spt"] == LIGHT.sequential
        assert colours["blend"] == LIGHT.sequential

    def test_everything_else_is_context_grey(self, tardiness):
        figure = policy_comparison(tardiness, LIGHT, highlight=["spt"])
        colours = dict(zip(figure.data[0].y, figure.data[0].marker.color))
        assert colours["mwkr"] == LIGHT.muted_mark
        assert colours["lpt"] == LIGHT.muted_mark

    def test_bars_are_sorted_by_value(self, tardiness):
        figure = policy_comparison(tardiness, LIGHT)
        values = list(figure.data[0].x)
        assert values == sorted(values, reverse=True)

    def test_the_bound_is_a_rule_not_a_series(self, tardiness):
        figure = policy_comparison(tardiness, LIGHT, lower_bound=529.0)
        assert len(figure.data) == 1  # still one bar trace
        assert figure.layout.shapes  # the bound is drawn as a shape

    def test_no_legend_for_a_single_series(self, tardiness):
        assert policy_comparison(tardiness, LIGHT).layout.showlegend is False

    def test_values_are_labelled_directly(self, tardiness):
        figure = policy_comparison(tardiness, LIGHT)
        assert figure.data[0].text is not None

    def test_labels_wear_text_tokens_not_series_colour(self, tardiness):
        figure = policy_comparison(tardiness, LIGHT)
        assert figure.data[0].textfont.color == LIGHT.text_secondary


class TestStationLoad:
    def test_bars_and_predicted_markers(self):
        actual = {"Machining": 0.59, "Inspection": 0.79}
        expected = {"Machining": 0.72, "Inspection": 0.87}
        figure = station_load(actual, LIGHT, expected)
        assert len(figure.data) == 2
        assert figure.layout.showlegend is True

    def test_legend_is_dropped_without_a_second_series(self):
        figure = station_load({"Machining": 0.59}, LIGHT)
        assert len(figure.data) == 1
        assert figure.layout.showlegend is False

    def test_axis_is_formatted_as_a_percentage(self):
        figure = station_load({"Machining": 0.59}, LIGHT)
        assert figure.layout.xaxis.tickformat == ".0%"


class TestLearningCurve:
    def test_it_is_a_single_series(self, history):
        figure = learning_curve(history, LIGHT)
        assert len(figure.data) == 1
        assert figure.layout.showlegend is False

    def test_parity_is_marked(self, history):
        assert learning_curve(history, LIGHT).layout.shapes

    def test_markers_are_large_enough_to_hit(self, history):
        assert learning_curve(history, LIGHT).data[0].marker.size >= 8


class TestNoDualAxis:
    """Two y-scales on one chart is the single worst chart mistake."""

    def test_no_figure_declares_a_second_axis(self, run, tardiness, history):
        figures = [
            gantt(run.completed, list(run.stations), LIGHT, max_jobs=10),
            policy_comparison(tardiness, LIGHT),
            station_load({"Machining": 0.5}, LIGHT),
            learning_curve(history, LIGHT),
        ]
        for figure in figures:
            assert "yaxis2" not in figure.layout


class TestDashboard:
    @pytest.fixture
    def page(self, run, tardiness, history):
        figures = {
            "gantt": (
                gantt(run.completed, list(run.stations), LIGHT, 10),
                gantt(run.completed, list(run.stations), DARK, 10),
            ),
            "policies": (
                policy_comparison(tardiness, LIGHT),
                policy_comparison(tardiness, DARK),
            ),
            "learning": (
                learning_curve(history, LIGHT),
                learning_curve(history, DARK),
            ),
        }
        return build_dashboard(
            figures=figures,
            tiles=[StatTile("On-time", "79%", "seed 1000")],
            family_counts={name: 3 for name in FAMILY_ORDER},
            subtitle="test page",
        )

    def test_it_renders_every_figure_in_both_modes(self, page):
        for slug in ("gantt", "policies", "learning"):
            assert f'id="{slug}-light"' in page
            assert f'id="{slug}-dark"' in page

    def test_no_python_placeholder_survives(self, page):
        """A stray brace in the CSS would break str.format silently."""
        for placeholder in ("{title}", "{body}", "{tiles}", "{ids}", "{mono}"):
            assert placeholder not in page

    def test_theme_switching_covers_all_three_states(self, page):
        assert "prefers-color-scheme: dark" in page
        assert ':root[data-theme="dark"]' in page
        assert ':root[data-theme="light"]' in page

    def test_the_family_table_provides_contrast_relief(self, page):
        """Three light slots fall under 3:1, which obliges a table view."""
        assert 'class="ftable"' in page
        for family in FAMILY_ORDER:
            assert family in page

    def test_scripts_come_from_an_allowlisted_cdn(self, page):
        assert "cdnjs.cloudflare.com" in page

    def test_stat_tiles_render(self, page):
        assert "On-time" in page and "79%" in page

    def test_it_is_valid_standalone_html(self, page):
        assert page.lstrip().startswith("<!doctype html>")
        assert page.rstrip().endswith("</html>")
