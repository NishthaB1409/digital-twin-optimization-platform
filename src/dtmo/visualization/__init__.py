"""Plotly figures and the HTML dashboard."""

from .charts import gantt, learning_curve, policy_comparison, station_load
from .dashboard import StatTile, build_dashboard
from .theme import DARK, FAMILY_ORDER, LIGHT, THEMES, Theme

__all__ = [
    "DARK",
    "FAMILY_ORDER",
    "LIGHT",
    "THEMES",
    "StatTile",
    "Theme",
    "build_dashboard",
    "gantt",
    "learning_curve",
    "policy_comparison",
    "station_load",
]
