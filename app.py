"""Streamlit app for the Lifespan Simulator."""

from __future__ import annotations

from math import inf

import pandas as pd
import streamlit as st

from lifespan_simulator.charts import (
    build_cause_projection_figure,
    build_survival_figure,
    build_total_rate_figure,
    build_trend_figure,
    format_metric,
    format_percentile_age,
)
from lifespan_simulator.data import (
    DATA_PATH,
    SIMULATOR_EXCLUDED_CAUSES,
    get_age_bucket_start,
    get_display_horizon,
    get_simulator_frame,
    load_mortality_data,
)
from lifespan_simulator.simulation import build_scenario_map, project_cause_rates, project_total_rates, simulate_survival
from lifespan_simulator.trends import build_trend_frame

st.set_page_config(page_title="Lifespan Simulator", layout="wide")


@st.cache_data(show_spinner=False)
def load_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = load_mortality_data(DATA_PATH)
    return raw, get_simulator_frame(raw)


def apply_custom_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(15, 118, 110, 0.10), transparent 28%),
                    radial-gradient(circle at top right, rgba(194, 65, 12, 0.10), transparent 26%),
                    linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
                color: #0f172a;
            }
            html, body, [class*="css"]  {
                font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
            }
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            div[data-testid="stMetric"] {
                background: rgba(255, 255, 255, 0.78);
                border: 1px solid rgba(148, 163, 184, 0.35);
                border-radius: 18px;
                padding: 0.8rem 1rem;
                box-shadow: 0 18px 40px rgba(15, 23, 42, 0.06);
            }
            div[data-baseweb="tab-list"] {
                gap: 0.5rem;
            }
            button[data-baseweb="tab"] {
                background: rgba(255, 255, 255, 0.7);
                border-radius: 999px;
                padding-left: 1rem;
                padding-right: 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def cause_options(frame: pd.DataFrame) -> dict[str, str]:
    return frame.drop_duplicates("cause_name").set_index("cause_name")["cause_code"].to_dict()


def cause_names_by_code(frame: pd.DataFrame) -> dict[str, str]:
    return frame.drop_duplicates("cause_code").set_index("cause_code")["cause_name"].to_dict()


def format_delta(baseline: float, scenario: float) -> str:
    if baseline == inf and scenario == inf:
        return "0.0"
    if baseline == inf:
        return "n/a"
    if scenario == inf:
        return "inf"
    return f"{scenario - baseline:+.1f}"


def format_optional_delta(baseline: float | None, scenario: float | None) -> str:
    if baseline is None or scenario is None:
        return "n/a"
    return format_delta(baseline, scenario)


def scenario_reference_age_band(simulator_frame: pd.DataFrame, initial_age: int) -> tuple[int, int, str, bool]:
    if initial_age <= 89:
        band_start, band_end = 85, 89
        is_fallback = False
    else:
        band_start = get_age_bucket_start(initial_age)
        band_end = 95 if band_start == 95 else band_start + 4
        is_fallback = True

    age_label = simulator_frame.loc[simulator_frame["age_start"] == band_start, "age_label"].iloc[0]
    return band_start, band_end, age_label, is_fallback


def scenario_top_causes(
    simulator_frame: pd.DataFrame,
    initial_age: int,
    scenario_map: dict[str, dict[str, float | int]],
    limit: int = 5,
) -> pd.DataFrame:
    cause_codes = simulator_frame.drop_duplicates("cause_code")["cause_code"].tolist()
    band_start, band_end, age_label, is_fallback = scenario_reference_age_band(simulator_frame, initial_age)
    projection = project_cause_rates(
        simulator_frame,
        initial_age,
        cause_codes,
        scenario_map,
        max_age=max(initial_age, band_end),
    )
    band_projection = projection[(projection["age"] >= band_start) & (projection["age"] <= band_end)].copy()
    ranked = (
        band_projection.groupby(["cause_code", "cause_name"], as_index=False)["scenario_rate"]
        .mean()
        .sort_values("scenario_rate", ascending=False)
        .head(limit)
        .reset_index(drop=True)
    )
    ranked.attrs["age_label"] = age_label
    ranked.attrs["is_fallback"] = is_fallback
    return ranked


def render_intro(simulator_frame: pd.DataFrame) -> None:
    st.title("Lifespan Simulator")
    st.caption(
        "Male mortality in Spain, 2011-2021. The simulator tab keeps the 2021 age-specific rates fixed unless "
        "you override selected causes in future calendar years."
    )
    with st.expander("Model assumptions used by the app", expanded=False):
        st.markdown(
            f"""
            - The simulator uses the 2021 age-specific mortality schedule only.
            - Mortality is treated as a constant annual hazard within each age bucket.
            - The `95 years or over` bucket remains flat after age 94.
            - The simulator works with {simulator_frame['cause_code'].nunique()} mutually exclusive cause series after excluding {len(SIMULATOR_EXCLUDED_CAUSES)} roll-up series that would otherwise double-count deaths.
            - The historical explorer keeps the full dataset, including aggregate categories, because it does not sum them.
            """
        )


def render_simulator_tab(simulator_frame: pd.DataFrame) -> None:
    all_cause_options = cause_options(simulator_frame)
    cause_name_lookup = cause_names_by_code(simulator_frame)
    age_min = int(simulator_frame["age_start"].min())
    age_max = int(simulator_frame["age_start"].max())

    controls_left, controls_right = st.columns([0.9, 1.9])
    with controls_left:
        initial_age = st.slider("Initial age", min_value=age_min, max_value=age_max, value=35, step=1)
    with controls_right:
        intervention_causes = st.multiselect(
            "Causes to modify in the scenario",
            options=sorted(all_cause_options.keys()),
            help="Set a future calendar year and replacement rate for each selected cause. Use 0 to model elimination.",
        )

    scenario_inputs: dict[str, dict[str, float | int]] = {}
    if intervention_causes:
        st.markdown("### Scenario controls")
        for cause_name in intervention_causes:
            cause_code = all_cause_options[cause_name]
            label_col, year_col, rate_col = st.columns([1.6, 1, 1])
            with label_col:
                st.markdown(f"**{cause_name}**")
            with year_col:
                effective_year = st.number_input(
                    f"Effective year for {cause_name}",
                    min_value=2021,
                    max_value=2120,
                    value=2030,
                    step=1,
                    key=f"year_{cause_code}",
                    label_visibility="collapsed",
                )
            with rate_col:
                replacement_rate = st.number_input(
                    f"Rate after intervention for {cause_name}",
                    min_value=0.0,
                    value=0.0,
                    step=0.1,
                    key=f"rate_{cause_code}",
                    label_visibility="collapsed",
                )
            scenario_inputs[cause_code] = {"effective_year": effective_year, "rate": replacement_rate}

    scenario_map = build_scenario_map(scenario_inputs)
    baseline_result = simulate_survival(simulator_frame, initial_age)
    scenario_result = simulate_survival(simulator_frame, initial_age, scenario_map)
    top_scenario_causes = scenario_top_causes(simulator_frame, initial_age, scenario_map, limit=5)
    top_scenario_names = top_scenario_causes["cause_name"].tolist()
    top_scenario_codes = top_scenario_causes["cause_code"].tolist()

    metrics = st.columns(4)
    metrics[0].metric("Baseline life expectancy", f"{format_metric(baseline_result.life_expectancy_age)} years")
    metrics[1].metric("Scenario life expectancy", f"{format_metric(scenario_result.life_expectancy_age)} years")
    metrics[2].metric(
        "Remaining years at start", format_metric(scenario_result.remaining_life_expectancy), format_delta(
            baseline_result.remaining_life_expectancy, scenario_result.remaining_life_expectancy
        )
    )
    metrics[3].metric(
        "Age at p50 death", format_percentile_age(scenario_result.percentiles["p50"]), format_optional_delta(
            baseline_result.percentiles["p50"], scenario_result.percentiles["p50"]
        )
    )

    horizon = get_display_horizon(initial_age)
    intervention_codes = [all_cause_options[name] for name in intervention_causes]
    visible_codes = list(dict.fromkeys([*top_scenario_codes, *intervention_codes]))

    chart_col, total_col = st.columns([1.45, 1])
    with chart_col:
        if visible_codes:
            cause_projection = project_cause_rates(simulator_frame, initial_age, visible_codes, scenario_map, horizon)
            st.plotly_chart(build_cause_projection_figure(cause_projection), use_container_width=True)
            if top_scenario_causes.attrs["is_fallback"]:
                st.caption(
                    f"The selected initial age is already above 89, so the chart ranks causes using "
                    f"{top_scenario_causes.attrs['age_label']} instead. Current leaders: {', '.join(top_scenario_names)}."
                )
            else:
                st.caption(
                    f"The chart shows the top 5 causes in the current scenario for "
                    f"{top_scenario_causes.attrs['age_label']}: {', '.join(top_scenario_names)}."
                )
        else:
            st.info("No cause series are available for the current scenario.")

    with total_col:
        total_projection = project_total_rates(simulator_frame, initial_age, scenario_map, horizon)
        st.plotly_chart(build_total_rate_figure(total_projection), use_container_width=True)

    survival_col, table_col = st.columns([1.35, 0.85])
    with survival_col:
        st.plotly_chart(
            build_survival_figure(baseline_result.survival_curve, scenario_result.survival_curve),
            use_container_width=True,
        )
        st.caption(
            "Percentiles refer to the modeled distribution of age at death for the synthetic cohort. "
            "For example, p75 is the age by which 75% of the cohort is expected to have died."
        )

    with table_col:
        percentile_rows = []
        for label in ["p50", "p75", "p90", "p95", "p99"]:
            baseline_value = baseline_result.percentiles[label]
            scenario_value = scenario_result.percentiles[label]
            percentile_rows.append(
                {
                    "Percentile": label,
                    "Baseline age": format_percentile_age(baseline_value),
                    "Scenario age": format_percentile_age(scenario_value),
                    "Scenario - baseline": (
                        "n/a"
                        if baseline_value is None or scenario_value is None
                        else f"{scenario_value - baseline_value:+.1f}"
                    ),
                }
            )
        st.markdown("### Survival percentiles")
        st.dataframe(pd.DataFrame(percentile_rows), hide_index=True, use_container_width=True)
        if scenario_map:
            st.markdown("### Active interventions")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Cause": cause_name_lookup[cause_code],
                            "Effective year": config["effective_year"],
                            "New rate": config["rate"],
                        }
                        for cause_code, config in scenario_map.items()
                    ]
                ),
                hide_index=True,
                use_container_width=True,
            )


def render_explorer_tab(raw_frame: pd.DataFrame) -> None:
    age_labels = raw_frame.drop_duplicates("age_start").sort_values("age_start")["age_label"].tolist()
    all_causes = raw_frame.drop_duplicates("cause_name").sort_values("cause_name")
    all_cause_names = all_causes["cause_name"].tolist()
    name_to_code = all_causes.set_index("cause_name")["cause_code"].to_dict()

    control_col, toggle_col = st.columns([1.4, 0.6])
    with control_col:
        age_label = st.selectbox("Age group", options=age_labels, index=0)
        default_causes = (
            raw_frame[(raw_frame["age_label"] == age_label) & (raw_frame["year"] == 2021)]
            .sort_values("rate", ascending=False)
            .head(3)["cause_name"]
            .tolist()
        )
        selected_causes = st.multiselect(
            "Causes of death",
            options=all_cause_names,
            default=default_causes,
            help="The explorer keeps the full source dataset, including aggregate categories.",
        )
    with toggle_col:
        log_scale = st.toggle("Log y-axis", value=False)

    if not selected_causes:
        st.info("Select at least one cause to explore 2011-2021 trends.")
        return

    trend_frames: dict[str, pd.DataFrame] = {}
    summary_rows = []

    for cause_name in selected_causes:
        cause_code = name_to_code[cause_name]
        subset = raw_frame[(raw_frame["age_label"] == age_label) & (raw_frame["cause_code"] == cause_code)].copy()
        trend_frame, stats = build_trend_frame(subset)
        trend_frames[cause_name] = trend_frame
        summary_rows.append(
            {
                "Cause": cause_name,
                "Slope per year": f"{stats['slope']:+.2f}",
                "R2": f"{stats['r_squared']:.3f}",
                "2021 rate": f"{trend_frame.loc[trend_frame['year'].idxmax(), 'rate']:.2f}",
            }
        )

    st.plotly_chart(build_trend_figure(trend_frames, age_label, log_scale), use_container_width=True)
    st.caption("Confidence intervals show the 95% interval around the fitted mean trend line, not a forecast interval.")
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)


def main() -> None:
    apply_custom_styles()
    raw_frame, simulator_frame = load_frames()
    render_intro(simulator_frame)
    simulator_tab, explorer_tab = st.tabs(["Simulator", "Mortality Rates Progress Explorer"])
    with simulator_tab:
        render_simulator_tab(simulator_frame)
    with explorer_tab:
        render_explorer_tab(raw_frame)


if __name__ == "__main__":
    main()
