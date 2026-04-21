"""Plotly chart builders used by the Streamlit app."""

from __future__ import annotations

from itertools import cycle
from math import inf
import re

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

PALETTE = [
    "#0f766e",
    "#c2410c",
    "#1d4ed8",
    "#be123c",
    "#4d7c0f",
    "#0f172a",
    "#7c3aed",
    "#b45309",
]
RGB_PATTERN = re.compile(r"rgba?\(([^)]+)\)")


def base_layout(title: str, yaxis_title: str) -> dict:
    return {
        "title": {"text": title, "x": 0.01},
        "template": "plotly_white",
        "height": 420,
        "hovermode": "x unified",
        "margin": {"l": 20, "r": 20, "t": 60, "b": 20},
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        "xaxis_title": "Attained age",
        "yaxis_title": yaxis_title,
    }


def build_cause_projection_figure(frame: pd.DataFrame) -> go.Figure:
    """Build the selected-cause projection chart."""
    fig = go.Figure()
    color_cycle = cycle(PALETTE)

    for cause_name, cause_frame in frame.groupby("cause_name", sort=False):
        color = next(color_cycle)
        fig.add_trace(
            go.Scatter(
                x=cause_frame["age"],
                y=cause_frame["scenario_rate"],
                mode="lines",
                line={"color": color, "width": 3},
                name=cause_name,
                legendgroup=cause_name,
                hovertemplate=(
                    f"Cause: {cause_name}<br>"
                    "Age %{x}<br>"
                    "Mortality rate: %{y:.2f} per 100k"
                    "<extra></extra>"
                ),
            )
        )

    layout = {
        "title": {"text": "Cause-Specific Mortality Paths", "x": 0.01, "yanchor": "bottom", "y": 0.95},
        "template": "plotly_white",
        "height": 420,
        "hovermode": "x unified",
        "margin": {"l": 20, "r": 20, "t": 60, "b": 20},
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 0.95, "xanchor": "left", "x": 0, "bgcolor": "rgba(0,0,0,0)"},
        "xaxis_title": "Attained age",
        "yaxis_title": "Deaths per 100,000",
    }

    fig.update_layout(layout)
    fig.update_layout(hovermode="closest")
    return fig


def build_total_rate_figure(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["age"],
            y=frame["baseline_rate"],
            mode="lines",
            name="Baseline total",
            line={"color": "#64748b", "dash": "dot", "width": 2},
            hovertemplate="Age %{x}<br>Baseline: %{y:.2f} per 100k<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["age"],
            y=frame["scenario_rate"],
            mode="lines",
            name="Scenario total",
            line={"color": "#0f172a", "width": 3},
            hovertemplate="Age %{x}<br>Scenario: %{y:.2f} per 100k<extra></extra>",
        )
    )
    fig.update_layout(base_layout("Total Mortality Rate", "Deaths per 100,000"))
    return fig


def build_survival_figure(baseline_curve: pd.DataFrame, scenario_curve: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=baseline_curve["age"],
            y=baseline_curve["survival"] * 100,
            mode="lines",
            name="Baseline survival",
            line={"color": "#94a3b8", "width": 3, "shape": "hv"},
            hovertemplate="Age %{x}<br>Baseline survival: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=scenario_curve["age"],
            y=scenario_curve["survival"] * 100,
            mode="lines",
            name="Scenario survival",
            line={"color": "#0f766e", "width": 3, "shape": "hv"},
            hovertemplate="Age %{x}<br>Scenario survival: %{y:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(base_layout("Synthetic Cohort Survival Curve", "Survival probability (%)"))
    fig.update_yaxes(range=[0, 100])
    return fig


def build_trend_figure(frames: dict[str, pd.DataFrame], age_label: str, log_scale: bool) -> go.Figure:
    fig = go.Figure()
    color_cycle = cycle(px.colors.qualitative.Safe)

    for cause_name, frame in frames.items():
        color = next(color_cycle)
        fig.add_trace(
            go.Scatter(
                x=frame["year"],
                y=frame["trend_upper"],
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
                legendgroup=cause_name,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=frame["year"],
                y=frame["trend_lower"],
                line={"width": 0},
                fill="tonexty",
                fillcolor=_color_with_alpha(color, 0.14),
                hoverinfo="skip",
                showlegend=False,
                legendgroup=cause_name,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=frame["year"],
                y=frame["rate"],
                mode="lines+markers",
                line={"color": color, "width": 2},
                marker={"size": 7},
                name=f"{cause_name} observed",
                legendgroup=cause_name,
                hovertemplate="Year %{x}<br>Observed: %{y:.2f} per 100k<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=frame["year"],
                y=frame["trend"],
                mode="lines",
                line={"color": color, "dash": "dash", "width": 3},
                name=f"{cause_name} trend",
                legendgroup=cause_name,
                hovertemplate="Year %{x}<br>Trend: %{y:.2f} per 100k<extra></extra>",
            )
        )

    fig.update_layout(
        {
            "title": {"text": f"Mortality Rate Progress for {age_label}", "x": 0.01},
            "template": "plotly_white",
            "height": 480,
            "hovermode": "x unified",
            "margin": {"l": 20, "r": 20, "t": 60, "b": 20},
            "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
            "xaxis_title": "Year",
            "yaxis_title": "Deaths per 100,000",
        }
    )
    if log_scale:
        fig.update_yaxes(type="log")
    return fig


def format_metric(value: float, digits: int = 1) -> str:
    if value == inf:
        return "inf"
    return f"{value:.{digits}f}"


def format_percentile_age(value: float | None) -> str:
    if value is None:
        return "Not reached"
    if value == inf:
        return "inf"
    return f"{value:.1f}"


def _color_with_alpha(color: str, alpha: float) -> str:
    color = color.strip()

    if color.startswith("#"):
        hex_value = color.lstrip("#")
        if len(hex_value) == 3:
            hex_value = "".join(channel * 2 for channel in hex_value)
        if len(hex_value) != 6:
            raise ValueError(f"Unsupported hex color: {color}")
        red, green, blue = (int(hex_value[index : index + 2], 16) for index in (0, 2, 4))
        return f"rgba({red}, {green}, {blue}, {alpha})"

    match = RGB_PATTERN.fullmatch(color)
    if match:
        channels = [part.strip() for part in match.group(1).split(",")]
        if len(channels) < 3:
            raise ValueError(f"Unsupported rgb color: {color}")
        red, green, blue = (int(float(channel)) for channel in channels[:3])
        return f"rgba({red}, {green}, {blue}, {alpha})"

    raise ValueError(f"Unsupported color format: {color}")
