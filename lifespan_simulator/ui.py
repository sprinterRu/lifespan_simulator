"""Streamlit presentation and widget state; numerical work lives in the model modules."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from math import inf, isfinite

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .analysis import compare_cause_totals, compare_life_expectancy, run_sensitivity
from .charts import (
    build_cause_projection_figure,
    build_survival_figure,
    build_total_rate_figure,
    build_trend_figure,
    format_metric,
    format_percentile_age,
)
from .config import (
    DEFAULT_INTEGRATION_END_AGE,
    MAX_INTEGRATION_END_AGE,
    PandemicConfig,
    ProjectionConfig,
)
from .data import (
    DEMENTIA_CODE,
    DEMENTIA_NAME,
    get_display_horizon,
    load_official_cause_totals,
    merge_dementia,
)
from .interventions import (
    FAST_NET_REDUCTION_DEFAULT,
    ScenarioMap,
    build_scenario_map,
    default_fast_rate,
)
from .mortality import MortalityModel
from .scenario_io import active_interventions_frame, parse_interventions_csv
from .simulation import project_cause_rates, project_total_rates, simulate_survival
from .tails import CauseTailFit, TailCurve, tail_fit_table
from .trends import TrendModel, build_trend_frame

EXPLORER_CAUSE_SELECTION_KEY = "explorer_selected_causes"
INTERVENTION_CAUSES_KEY = "simulator_intervention_causes"
SLOW_START_YEAR_DEFAULT = 2040
SLOW_RATE_DEFAULT = 0.015
FAST_START_YEAR_DEFAULT = 2070
FLOOR_DEFAULT = 0.01


def apply_custom_styles() -> None:
    """Inject app-wide CSS used by Streamlit when rendering the page."""
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
    """Build the display-name-to-code lookup used by simulator controls."""
    return frame.drop_duplicates("cause_name").set_index("cause_name")["cause_code"].to_dict()


def cause_names_by_code(frame: pd.DataFrame) -> dict[str, str]:
    """Build the code-to-display-name lookup used by tables."""
    return frame.drop_duplicates("cause_code").set_index("cause_code")["cause_name"].to_dict()


def apply_loaded_interventions(
    cause_names: list[str],
    scenario_inputs: dict[str, dict[str, float | int]],
    all_cause_options: dict[str, str],
) -> None:
    """Seed Streamlit widget state from a validated intervention CSV."""
    st.session_state[INTERVENTION_CAUSES_KEY] = cause_names
    for cause_name in cause_names:
        cause_code = all_cause_options[cause_name]
        config = scenario_inputs[cause_code]
        st.session_state[f"slow_start_year_{cause_code}"] = int(config["slow_start_year"])
        st.session_state[f"slow_rate_{cause_code}"] = float(config["slow_rate"])
        st.session_state[f"fast_start_year_{cause_code}"] = int(config["fast_start_year"])
        st.session_state[f"fast_rate_{cause_code}"] = float(config["fast_rate"])
        st.session_state[f"fast_rate_custom_{cause_code}"] = True
        st.session_state[f"floor_{cause_code}"] = float(config["floor"])


def default_explorer_causes(raw_frame: pd.DataFrame, age_label: str) -> list[str]:
    """Choose initial cause selections for the historical explorer."""
    return (
        raw_frame[(raw_frame["age_label"] == age_label) & (raw_frame["year"] == 2021)]
        .sort_values("rate", ascending=False)
        .head(3)["cause_name"]
        .tolist()
    )


def format_delta(baseline: float | None, scenario: float | None) -> str | None:
    """Omit an unavailable metric delta so Streamlit does not draw a gain arrow."""
    if baseline is None or scenario is None:
        return None
    if not isfinite(baseline) or not isfinite(scenario):
        return None
    return f"{scenario - baseline:+.1f}"


def format_optional_delta(baseline: float | None, scenario: float | None) -> str:
    """Use a text placeholder for unavailable deltas in the percentile table."""
    return format_delta(baseline, scenario) or "n/a"


def format_slope_ci(stats: dict[str, object]) -> str:
    """Format the 95% confidence interval for an explorer trend slope."""
    lower = stats.get("slope_ci_lower")
    upper = stats.get("slope_ci_upper")
    if pd.isna(lower) or pd.isna(upper):
        return "n/a"
    return f"[{float(lower):+.2f}, {float(upper):+.2f}]"


def format_slope_significance(stats: dict[str, object]) -> str:
    """Describe whether the slope confidence interval excludes zero."""
    lower = stats.get("slope_ci_lower")
    upper = stats.get("slope_ci_upper")
    if pd.isna(lower) or pd.isna(upper):
        return "n/a"
    return "Yes" if bool(stats.get("slope_significant")) else "No"


def available_top_cause_age_bands(simulator_frame: pd.DataFrame, initial_age: int) -> pd.DataFrame:
    """Filter age bands that can be used to rank top scenario causes."""
    age_bands = (
        simulator_frame.drop_duplicates("age_start")
        .sort_values("age_start")[["age_start", "age_end", "age_label"]]
        .reset_index(drop=True)
    )
    band_ends = age_bands["age_end"].fillna(age_bands["age_start"])
    eligible = age_bands[band_ends >= initial_age]
    return (eligible if not eligible.empty else age_bands.tail(1)).reset_index(drop=True)


def default_top_cause_age_start(age_starts: list[int]) -> int:
    """Pick the default age bucket for ranking top causes."""
    return 85 if 85 in age_starts else age_starts[0]


def age_band_details(simulator_frame: pd.DataFrame, age_start: int) -> tuple[int, int, str]:
    """Return numeric bounds and label for one simulator age bucket."""
    age_band = simulator_frame.loc[simulator_frame["age_start"] == age_start].iloc[0]
    age_end = age_start if pd.isna(age_band["age_end"]) else int(age_band["age_end"])
    return age_start, age_end, str(age_band["age_label"])


def render_explorer_tab(raw_frame: pd.DataFrame) -> None:
    """Render the historical mortality-rate trend explorer."""
    raw_frame = raw_frame[raw_frame.age_start < 95]
    age_labels = (
        raw_frame.drop_duplicates("age_start").sort_values("age_start")["age_label"].tolist()
    )
    all_causes = raw_frame.drop_duplicates("cause_name").sort_values("cause_name")
    all_cause_names = all_causes["cause_name"].tolist()
    name_to_code = all_causes.set_index("cause_name")["cause_code"].to_dict()

    control_col, toggle_col = st.columns([1.4, 0.6])
    with control_col:
        age_label = st.selectbox("Age group", options=age_labels, index=0)
        # Seed the explorer with default causes only once; otherwise Streamlit's
        # rerun after changing age group would replace the user's selection.
        if EXPLORER_CAUSE_SELECTION_KEY not in st.session_state:
            st.session_state[EXPLORER_CAUSE_SELECTION_KEY] = default_explorer_causes(
                raw_frame, age_label
            )
        selected_causes = st.multiselect(
            "Causes of death",
            options=all_cause_names,
            help="The explorer keeps the full source dataset, including aggregate categories.",
            key=EXPLORER_CAUSE_SELECTION_KEY,
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
        subset = raw_frame[
            (raw_frame["age_label"] == age_label) & (raw_frame["cause_code"] == cause_code)
        ].copy()
        trend_frame, stats = build_trend_frame(subset)
        trend_frames[cause_name] = trend_frame
        summary_rows.append(
            {
                "Cause": cause_name,
                "Slope per year": f"{stats['slope']:+.2f}",
                "Slope 95% CI": format_slope_ci(stats),
                "Slope excludes 0": format_slope_significance(stats),
                "R2": f"{stats['r_squared']:.3f}",
                "2021 rate": f"{trend_frame.loc[trend_frame['year'].idxmax(), 'rate']:.2f}",
            }
        )

    st.plotly_chart(build_trend_figure(trend_frames, age_label, log_scale), width="stretch")
    st.caption(
        "Chart bands show the 95% interval around the fitted mean trend line, not a forecast interval. "
        "A slope is marked as excluding 0 when its 95% confidence interval does not include zero."
    )
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, width="stretch")


def mark_fast_rate_custom(cause_code: str) -> None:
    """A user edit is authoritative, even when it equals a previous default."""
    st.session_state[f"fast_rate_custom_{cause_code}"] = True


def prepare_fast_rate(cause_code: str, suggested: float) -> str:
    """Refresh automatic defaults as shapes change; preserve edits and imports."""
    key = f"fast_rate_{cause_code}"
    custom_key = f"fast_rate_custom_{cause_code}"
    if key not in st.session_state:
        st.session_state[key] = suggested
        st.session_state[custom_key] = False
    elif custom_key not in st.session_state:
        # Existing sessions predate provenance tracking; do not guess whether
        # their value was edited. The explicit reset button adopts new defaults.
        st.session_state[custom_key] = True
    elif not st.session_state[custom_key]:
        st.session_state[key] = suggested
    return key


def render_interventions(
    intervention_causes: list[str],
    all_cause_options: dict[str, str],
    suggested_fast_rates: dict[str, float],
) -> ScenarioMap:
    """Edit intervention parameters, keeping CSV imports and widgets in sync."""
    scenario_inputs: dict[str, dict[str, float | int]] = {}
    if intervention_causes:
        st.markdown("### Scenario controls")
        header_cols = st.columns([1.7, 0.95, 1.0, 0.95, 1.0, 0.8])
        for column, label in zip(
            header_cols,
            [
                "Cause",
                "Slow start",
                "Slow reduction",
                "Fast start",
                "Fast reduction",
                "Slow floor",
            ],
            strict=True,
        ):
            column.caption(label)
        st.caption(
            "Slow floor limits only gradual improvements. Fast compounds on the attained slow multiplier "
            "without a floor. Values are reductions relative to the untreated rate; 1 eliminates the cause. "
            "Exponential-tail defaults offset ageing plus 10% net reduction per year. Other families default to 10%. "
            "The net target applies after age 94 with calendar factors held fixed."
        )
        for cause_name in intervention_causes:
            cause_code = all_cause_options[cause_name]
            label_col, slow_year_col, slow_rate_col, fast_year_col, fast_rate_col, floor_col = (
                st.columns([1.7, 0.95, 1.0, 0.95, 1.0, 0.8])
            )
            with label_col:
                st.markdown(f"**{cause_name}**")
            with slow_year_col:
                slow_start_year = st.number_input(
                    f"Slow start year for {cause_name}",
                    min_value=2021,
                    max_value=2120,
                    value=SLOW_START_YEAR_DEFAULT,
                    step=1,
                    key=f"slow_start_year_{cause_code}",
                    label_visibility="collapsed",
                )
            with slow_rate_col:
                slow_rate = st.number_input(
                    f"Slow annual reduction for {cause_name}",
                    min_value=0.0,
                    max_value=1.0,
                    value=SLOW_RATE_DEFAULT,
                    step=0.005,
                    format="%.3f",
                    key=f"slow_rate_{cause_code}",
                    label_visibility="collapsed",
                )
            fast_year_key = f"fast_start_year_{cause_code}"
            if st.session_state.get(fast_year_key, FAST_START_YEAR_DEFAULT) < slow_start_year:
                st.session_state[fast_year_key] = int(slow_start_year)
            with fast_year_col:
                fast_start_year = st.number_input(
                    f"Fast start year for {cause_name}",
                    min_value=int(slow_start_year),
                    max_value=2120,
                    value=max(FAST_START_YEAR_DEFAULT, int(slow_start_year)),
                    step=1,
                    key=fast_year_key,
                    label_visibility="collapsed",
                )
            with fast_rate_col:
                suggested = suggested_fast_rates[cause_code]
                fast_rate_key = prepare_fast_rate(cause_code, suggested)
                fast_rate = st.number_input(
                    f"Fast annual reduction for {cause_name}",
                    min_value=0.0,
                    max_value=1.0,
                    value=None,
                    step=0.01,
                    format="%.6f",
                    key=fast_rate_key,
                    on_change=mark_fast_rate_custom,
                    args=(cause_code,),
                    help=f"Suggested for the active age curve: {suggested:.2%}. Manual and imported values are preserved. 0.1 means a 10% treatment reduction before age growth.",
                    label_visibility="collapsed",
                )
                if fast_rate is None:
                    st.error(f"Enter a fast annual reduction for {cause_name}.")
                    st.stop()
            with floor_col:
                floor = st.number_input(
                    f"Slow-stage floor for {cause_name}",
                    min_value=1e-12,
                    max_value=1.0,
                    value=FLOOR_DEFAULT,
                    step=0.001,
                    format="%.4f",
                    key=f"floor_{cause_code}",
                    help="Minimum multiplier from slow improvements only. Fast can reduce risk below this value.",
                    label_visibility="collapsed",
                )
            scenario_inputs[cause_code] = {
                "slow_start_year": slow_start_year,
                "slow_rate": slow_rate,
                "fast_start_year": fast_start_year,
                "fast_rate": fast_rate,
                "floor": floor,
            }

    return build_scenario_map(scenario_inputs)


def render_projection_controls() -> tuple[int, bool, ProjectionConfig]:
    left, middle, right = st.columns(3)
    initial_age = left.slider("Initial age", 35, 95, 35)
    start_year = middle.number_input("Cohort start year", 2021, 2120, 2021)
    use_trends = right.toggle("Use historical trends (2011–2019)", value=True)
    with st.expander("Mortality assumptions", expanded=False):
        left, right = st.columns(2)
        with left:
            calibrate = st.toggle(
                "Calibrate to official all-cause mortality",
                value=True,
                help="Preserves cause shares and matches Eurostat single-age Mx for 2021 at ages 35–94.",
            )
            old_age = st.selectbox(
                "Mortality after age 94",
                ["individual", "kannisto", "gompertz", "flat"],
                format_func=lambda x: {
                    "individual": "Individual cause trajectories (fitted)",
                    "kannisto": "Kannisto (logistic)",
                    "gompertz": "Gompertz (exponential)",
                    "flat": "Flat after 94 (comparison only)",
                }[x],
            )
            pandemic_mode = st.selectbox(
                "COVID path",
                ["fade", "hold_2021", "exclude"],
                format_func=lambda x: {
                    "fade": "Fade toward a residual level",
                    "hold_2021": "Hold 2021 level",
                    "exclude": "Exclude COVID (counterfactual)",
                }[x],
            )
            half_life = st.number_input(
                "COVID half-life (years)", 0.1, 50.0, 3.0, disabled=pandemic_mode != "fade"
            )
            residual = st.slider(
                "COVID residual share of 2021 mortality",
                0.0,
                1.0,
                0.1,
                0.05,
                disabled=pandemic_mode != "fade",
            )
        with right:
            decline_residual = st.slider(
                "Declining-tail residual share",
                0.0,
                0.5,
                0.0,
                0.05,
                disabled=old_age != "individual",
                help="Baseline declining curves approach this share of their age-94 rate. Fast interventions can still reduce it without a floor.",
            )
            trend_floor = st.number_input(
                "Trend floor", 0.0, 1.0, 0.15, 0.05, disabled=not use_trends
            )
            trend_ceiling = st.number_input(
                "Trend ceiling", 1.0, 10.0, 2.0, 0.25, disabled=not use_trends
            )
            trend_horizon = st.slider(
                "Years of trend extrapolation after 2021", 0, 100, 50, disabled=not use_trends
            )
            max_age = st.number_input(
                "Numerical integration end age",
                min_value=100,
                max_value=MAX_INTEGRATION_END_AGE,
                value=DEFAULT_INTEGRATION_END_AGE,
                step=100,
                help="A computational horizon, not a maximum human lifespan. Default 2,000 years; increase up to 10,000 for longer-lived scenarios. The remaining-tail check still applies.",
            )
        st.caption(
            "The COVID fade parameters are assumptions, not estimates of future outbreaks. "
            "Non-COVID 2021 rates may still contain indirect pandemic effects. "
            "Beyond the trend horizon, time trends stop changing; ageing continues."
        )
    return (
        initial_age,
        use_trends,
        ProjectionConfig(
            start_year=int(start_year),
            calibrate=calibrate,
            old_age_model=old_age,
            decline_residual_fraction=decline_residual,
            pandemic=PandemicConfig(pandemic_mode, half_life, residual),
            trend_floor=trend_floor,
            trend_ceiling=trend_ceiling,
            trend_horizon_years=trend_horizon if use_trends else 0,
            max_age=max_age,
        ),
    )


def render_simulator_tab(
    frame: pd.DataFrame,
    official: pd.DataFrame,
    trend: TrendModel,
    tails: dict[str, CauseTailFit],
) -> tuple[dict[str, CauseTailFit], ProjectionConfig]:
    initial_age, use_trends, config = render_projection_controls()
    active_tails = render_tail_overrides(frame, tails, config.old_age_model == "individual")
    model = MortalityModel(frame, trend.slope_table if use_trends else None, official, active_tails)
    if config.old_age_model == "individual":
        st.caption(
            "Each cause follows its own fitted tail after 94. See Cause trajectories for families, "
            "validation errors and fallback assumptions. Dementia includes Alzheimer disease."
        )
    if use_trends:
        st.caption(
            f"Using {trend.significant_count} of {trend.total_count} non-COVID cause-age trends fitted through 2019. "
            "The baseline is conditional on survival to the selected starting age and includes the chosen COVID path."
        )
    options = cause_options(frame)
    exponential_growth = model.exponential_age_growth(config)
    suggested_fast_rates = {
        code: default_fast_rate(exponential_growth.get(code, 0.0)) for code in model.codes
    }
    with st.expander("Import intervention settings"):
        uploaded = st.file_uploader("Load active interventions CSV", type=["csv"])
        if st.button("Load interventions", disabled=uploaded is None):
            try:
                names, inputs = parse_interventions_csv(uploaded, options)
                apply_loaded_interventions(names, inputs, options)
                st.success(f"Loaded {len(names)} intervention(s).")
            except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as error:
                st.error(f"Could not load interventions CSV. {error}")
    previous = st.session_state.get(INTERVENTION_CAUSES_KEY, [])
    if any(name not in options for name in previous):
        st.session_state[INTERVENTION_CAUSES_KEY] = [name for name in previous if name in options]
        st.warning(
            "The cause list changed. Dementia and Alzheimer disease now share one intervention; "
            "select the combined group to set its parameters."
        )
    if st.button(
        "Select all exponentially growing causes",
        disabled=not exponential_growth,
        help="Adds every cause with a positive exponential slope in the active model, including custom trajectories. Existing selections are kept.",
    ):
        st.session_state[INTERVENTION_CAUSES_KEY] = sorted(
            set(st.session_state.get(INTERVENTION_CAUSES_KEY, []))
            | {model.names[code] for code in exponential_growth}
        )
    selected = st.multiselect(
        "Causes to modify in the scenario", sorted(options), key=INTERVENTION_CAUSES_KEY
    )
    if st.button("Reset selected fast rates to defaults", disabled=not selected):
        for name in selected:
            code = options[name]
            st.session_state[f"fast_rate_{code}"] = suggested_fast_rates[code]
            st.session_state[f"fast_rate_custom_{code}"] = False
    scenarios = render_interventions(selected, options, suggested_fast_rates)
    baseline = simulate_survival(model, initial_age, config=config)
    scenario = simulate_survival(model, initial_age, scenarios, config) if scenarios else baseline

    metrics = st.columns(4)
    metrics[0].metric("Baseline mean age at death", format_metric(baseline.life_expectancy_age))
    metrics[1].metric(
        "Scenario mean age at death",
        format_metric(scenario.life_expectancy_age),
        format_delta(baseline.life_expectancy_age, scenario.life_expectancy_age),
        help="Unresolved means more tail integration is needed. ∞ means the model analytically retains positive survival forever, so its full mean is infinite. See the diagnostics below.",
    )
    metrics[2].metric("Scenario remaining years", format_metric(scenario.remaining_life_expectancy))
    metrics[3].metric("Scenario median age", format_percentile_age(scenario.percentiles["p50"]))
    if not baseline.mean_resolved or not scenario.mean_resolved:
        st.warning(
            f"Simulation completed, but the full mean is not resolved by age {config.max_age}. "
            "Survivors beyond that age could still materially affect the mean under the selected assumptions. "
            "The median can still be known: it only requires half of the cohort to have died. "
            "See the survival percentages and numerical bounds below. Increasing the horizon may narrow "
            "the bounds, but does not guarantee a resolved mean."
        )
    diagnostics = []
    for label, result in [("Baseline", baseline), ("Scenario", scenario)]:
        if result.infinite_mean:
            st.info(
                f"{label}: the full mean is infinite under this model. "
                "All remaining mortality terms eventually decay fast enough to leave positive limiting survival. "
                "A longer horizon cannot produce a finite mean. Restricted life-years and reached percentiles "
                "remain available; this is a consequence of the scenario assumptions."
            )
        lower_mean_age = initial_age + result.restricted_remaining_life_expectancy
        upper_mean_age = lower_mean_age + result.remaining_expectation_bound
        if not result.converged:
            st.caption(
                f"{label}: {100 * result.survival_at_end:.2f}% of the modeled cohort is still alive "
                f"at age {config.max_age}. Expected remaining years lived before that age: "
                f"{result.restricted_remaining_life_expectancy:.2f}."
            )
        diagnostics.append(
            {
                "Projection": label,
                f"Expected remaining years lived before {config.max_age}": result.restricted_remaining_life_expectancy,
                f"Alive at {config.max_age} (%)": 100 * result.survival_at_end,
                "Unintegrated life-years upper bound": result.remaining_expectation_bound,
                "Mean age at death: lower bound": lower_mean_age,
                "Mean age at death: upper bound": upper_mean_age
                if isfinite(upper_mean_age)
                else None,
                "Limiting survival: lower bound (%)": 100 * result.survival_forever_lower_bound
                if result.survival_forever_lower_bound is not None
                else None,
                "Mean status": "Infinite"
                if result.infinite_mean
                else "Resolved"
                if result.converged
                else "Unresolved",
                "Full mean resolved": result.mean_resolved,
            }
        )
    with st.expander(
        "Numerical horizon and remaining survival",
        expanded=not baseline.converged or not scenario.converged,
    ):
        st.dataframe(pd.DataFrame(diagnostics), hide_index=True)
        st.caption(
            "Expected years before the horizon is a restricted mean. Adding the initial age gives the "
            "lower bound on mean age at death; adding the omitted life-years bound gives the upper bound. "
            "A blank upper mean bound means no finite bound was established, not proof of an infinite mean. "
            "A finite resolved mean has at most "
            f"{config.expectation_tolerance:g} years of omitted tail contribution under the selected assumptions. "
            "These are numerical bounds within the model, not confidence intervals or bounds on forecast uncertainty."
            " Limiting survival bounds apply only when an infinite mean has been established analytically."
        )

    age_bands = available_top_cause_age_bands(frame, initial_age)
    age_starts = age_bands.age_start.astype(int).tolist()
    labels = age_bands.set_index("age_start").age_label.to_dict()
    ranking_age = st.selectbox(
        "Rank top causes by age group",
        age_starts,
        index=age_starts.index(default_top_cause_age_start(age_starts)),
        format_func=labels.get,
    )
    band_start, band_end, _ = age_band_details(frame, ranking_age)
    if initial_age > band_end:
        st.caption(f"Ranking uses extrapolated rates at the initial age {initial_age}.")
    horizon = max(get_display_horizon(initial_age), band_end)
    all_rates = project_cause_rates(
        model, initial_age, list(model.codes), scenarios, horizon, config
    )
    ranked = (
        all_rates[all_rates.age.between(max(initial_age, band_start), max(initial_age, band_end))]
        .groupby("cause_code")
        .scenario_rate.mean()
        .nlargest(5)
        .index.tolist()
    )
    visible = set(ranked) | set(scenarios)
    left, right = st.columns([1.4, 1])
    with left:
        st.plotly_chart(
            build_cause_projection_figure(all_rates[all_rates.cause_code.isin(visible)]),
            width="stretch",
        )
    with right:
        st.plotly_chart(
            build_total_rate_figure(
                project_total_rates(model, initial_age, scenarios, horizon, config)
            ),
            width="stretch",
        )
    st.plotly_chart(
        build_survival_figure(baseline.survival_curve, scenario.survival_curve), width="stretch"
    )
    st.caption(
        "The curve includes the full numerical horizon; use the range slider to explore the tail. "
        "The initial view covers at least 99% of deaths when that percentile is reached. "
        "p75 is the age by which 75% of the cohort has died; "
        "these percentiles describe individual lifespans under one model, not confidence in the forecast."
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Percentile": key,
                    "Baseline age": format_percentile_age(value),
                    "Scenario age": format_percentile_age(scenario.percentiles[key]),
                    "Change": format_optional_delta(value, scenario.percentiles[key]),
                }
                for key, value in baseline.percentiles.items()
            ]
        ),
        hide_index=True,
    )
    if scenarios:
        interventions = active_interventions_frame(scenarios, model.names)
        st.download_button(
            "Download active interventions CSV",
            interventions.to_csv(index=False),
            "interventions.csv",
            "text/csv",
        )
    assumptions = {
        "initial_age": initial_age,
        "projection": asdict(config),
        "interventions": {code: value._asdict() for code, value in scenarios.items()},
        "intervention_floor_applies_to": "slow stage only",
        "fast_default_policy": {
            "net_annual_reduction_for_exponential_tails": FAST_NET_REDUCTION_DEFAULT,
            "formula": "1 - (1 - net_reduction) * exp(-age_log_growth)",
            "suggested_rates": suggested_fast_rates,
            "scope": "Active exponential age tails after 94; calendar trends and COVID changes are not compensated.",
        },
        "historical_trends": "2011–2019" if use_trends else "disabled",
        "reference_population": "Spain, males, 2021",
        "cause_tail_fits": {code: fit.record() for code, fit in model.tail_fits.items()},
    }
    st.download_button(
        "Download model assumptions JSON",
        json.dumps(assumptions, indent=2),
        "assumptions.json",
        "application/json",
    )
    render_sensitivity(model, initial_age, scenarios, config)
    return active_tails, config


def render_tail_overrides(
    frame: pd.DataFrame, fitted: dict[str, CauseTailFit], enabled: bool
) -> dict[str, CauseTailFit]:
    """Keep user scenarios separate from cached, automatic fit results."""
    active = dict(fitted)
    with st.expander("Override individual cause trajectories"):
        st.caption(
            "Optional scenario assumptions. Overrides also appear in Cause trajectories and exports. "
            "Their validation scores are cleared because they are no longer the automatic fit."
        )
        dementia = fitted.get(DEMENTIA_CODE)
        if dementia is not None and dementia.alternative_curve is not None:
            choice = st.radio(
                "Dementia age trajectory",
                ["Exponent fitted at 70–89 (default)", "Sigmoid fitted at 70–94 (comparison)"],
                disabled=not enabled,
                help="The default extrapolates the younger-age trend. The sigmoid tests a saturation assumption.",
            )
            if enabled and choice.startswith("Sigmoid"):
                active[DEMENTIA_CODE] = replace(
                    dementia,
                    curve=dementia.alternative_curve,
                    fit_ages=dementia.alternative_fit_ages,
                    status="Comparison scenario",
                    reason="Sigmoid fitted at 70–94. Its better near-age predictions do not establish a remote plateau.",
                    validation_log_rmse=dementia.alternative_log_rmse,
                    alternative=dementia.curve.kind,
                    alternative_curve=dementia.curve,
                    alternative_fit_ages=dementia.fit_ages,
                    alternative_log_rmse=None,
                )
        options = cause_options(frame)
        selected = st.multiselect(
            "Causes with custom age trajectories",
            sorted(options),
            disabled=not enabled,
        )
        for name in selected:
            code = options[name]
            original = active[code]
            families = ["exponent", "sigmoid", "plateau", "decline"]
            left, middle, right = st.columns(3)
            family = left.selectbox(
                f"Tail family for {name}",
                families,
                index=families.index(original.curve.kind),
                key=f"tail_family_{code}_{original.curve.kind}",
                disabled=not enabled,
            )
            initial_curve = original.curve
            if original.alternative_curve is not None and original.alternative_curve.kind == family:
                initial_curve = original.alternative_curve
            parameter_key = f"{code}_{original.curve.kind}_{family}"
            slope = middle.number_input(
                f"Tail slope for {name}",
                min_value=0.0,
                max_value=1.0,
                value=float(initial_curve.slope),
                step=0.01,
                key=f"tail_slope_{parameter_key}",
                disabled=not enabled or family == "plateau",
                help="Exponent/decline: logarithmic rate change per year. Sigmoid: logistic shape slope.",
            )
            ratio = right.number_input(
                f"Plateau / age-94 rate for {name}",
                min_value=1.0,
                max_value=1000.0,
                value=float(initial_curve.asymptote_ratio)
                if initial_curve.kind == "sigmoid"
                else 2.0,
                step=0.5,
                key=f"tail_plateau_{parameter_key}",
                disabled=not enabled or family != "sigmoid",
            )
            if enabled:
                active[code] = replace(
                    original,
                    curve=TailCurve(family, slope, ratio),
                    fit_ages="manual",
                    status="Manual override",
                    reason="User-specified tail shape; not selected by validation.",
                    validation_log_rmse=None,
                    year_agreement=None,
                    alternative=original.curve.kind,
                    alternative_curve=original.curve,
                    alternative_fit_ages=original.fit_ages,
                    alternative_log_rmse=original.validation_log_rmse,
                )
    return active


def render_sensitivity(
    model: MortalityModel, initial_age: int, scenarios: ScenarioMap, config: ProjectionConfig
) -> None:
    st.subheader("Sensitivity to assumptions")
    st.caption(
        "Each row changes one assumption and recomputes baseline and intervention together. "
        "The range is a scenario comparison, not a confidence interval. Other assumptions stay as selected above."
    )
    if not st.checkbox("Run sensitivity analysis"):
        return
    with st.spinner("Comparing assumptions…"):
        results = run_sensitivity(model, initial_age, scenarios, config)
    figure = go.Figure()
    for column, color in [("Baseline mean age", "#94a3b8"), ("Scenario mean age", "#0f766e")]:
        figure.add_trace(
            go.Bar(
                y=results.Assumption,
                x=results[column].replace([inf, -inf], float("nan")),
                name=column,
                orientation="h",
                marker_color=color,
            )
        )
    figure.update_layout(
        barmode="group",
        height=max(480, len(results) * 34),
        xaxis_title="Mean age at death (finite, resolved projections only)",
        yaxis={"autorange": "reversed"},
        template="plotly_white",
    )
    st.plotly_chart(figure, width="stretch")
    st.dataframe(results, hide_index=True)
    st.caption(
        f"Blank means/percentiles extend beyond the age-{config.max_age} calculation or fail its remainder bound. "
        "Infinite means are identified analytically and appear in the table, not as bars. "
        "Restricted expected years use the same horizon in every row. Slope variations have no effect in flat mode; "
        "trend variations have no effect when trends are disabled."
    )
    st.download_button(
        "Download sensitivity CSV", results.to_csv(index=False), "sensitivity.csv", "text/csv"
    )


def render_validation_tab(
    frame: pd.DataFrame,
    official: pd.DataFrame,
    tails: dict[str, CauseTailFit],
    config: ProjectionConfig,
) -> None:
    st.subheader("Reference checks: Spain, males, 2021")
    st.markdown(
        "Reference snapshots: [Eurostat life tables](https://ec.europa.eu/eurostat/databrowser/view/demo_mlifetable/default/table) "
        "and [all-cause mortality](https://ec.europa.eu/eurostat/databrowser/view/hlth_cd_acdr2/default/table). "
        "Sources, retrieval date, units and checksums are recorded in `data/reference_metadata.json`."
    )
    st.caption(
        "These period-table checks always disable future trends and hold COVID at its observed 2021 level. "
        "They do not validate a future cohort forecast. Life-table and cause-of-death rates have different production methods."
    )
    audit = compare_cause_totals(frame, load_official_cause_totals())
    st.markdown("**Source cause totals before calibration**")
    st.dataframe(
        audit[["age_display", "rate_csv", "rate_official", "difference", "difference_pct"]].rename(
            columns={
                "age_display": "Age group",
                "rate_csv": "Source rate",
                "rate_official": "Official rate",
                "difference": "Difference per 100,000",
                "difference_pct": "Difference (%)",
            }
        ),
        hide_index=True,
    )
    st.caption(
        "Rates are deaths per 100,000. Differences remain visible: calibration does not establish why the source differs."
    )
    expectancy = compare_life_expectancy(frame, official, tails, config.decline_residual_fraction)
    st.markdown("**Remaining life expectancy at selected exact ages**")
    st.dataframe(
        expectancy.rename(
            columns={
                "age": "Starting age",
                "official_remaining_years": "Official years",
                "uncalibrated_flat": "Source + flat after 94",
                "uncalibrated_kannisto": "Source + Kannisto",
                "calibrated_kannisto": "Calibrated + Kannisto",
                "calibrated_difference": "Difference (years)",
                "calibrated_individual": "Calibrated + individual tails",
                "individual_difference": "Individual-tail difference (years)",
            }
        ),
        hide_index=True,
    )
    st.caption(
        "Official life expectancy is published to 0.1 years. Mx is interpreted as a constant hazard within each year. "
        "All 95+ observations are excluded. Kannisto is fitted to ages 80–94 and extrapolated beyond 94."
    )
    fit = MortalityModel(frame, official_life_table=official).old_age_fit(ProjectionConfig())
    with st.expander("Old-age fit and limitations"):
        st.json(fit)
        st.markdown(
            "The fit uses least squares on logit mortality, an asymptote of 1/year, and an anchor at age 94. "
            "It is inspired by the [HMD methodology](https://www.mortality.org/File/GetDocument/Public/Docs/MethodsProtocolV6.pdf), "
            "but does not reproduce its exposure-weighted estimation. This is the legacy common-curve comparator; "
            "individual mode fits separate tails and allows cause shares to change. "
            "Interventions can reduce the old-age hazard, so extreme scenarios can still have very long lifespans."
        )
    st.download_button(
        "Download life-table comparison CSV",
        expectancy.to_csv(index=False),
        "reference_comparison.csv",
        "text/csv",
    )


def render_trajectories_tab(
    raw: pd.DataFrame,
    frame: pd.DataFrame,
    official: pd.DataFrame,
    tails: dict[str, CauseTailFit],
    config: ProjectionConfig,
) -> None:
    st.subheader("Individual late-age trajectories")
    st.caption(
        "Shapes use 2011–2019; levels use 2021. Exponent, sigmoid, plateau and decline "
        "are spliced continuously at age 94. Dementia defaults to an exponent fitted at 70–89; "
        "its 70–94 sigmoid is a comparison. All 95+ observations are excluded."
    )
    names = cause_names_by_code(frame)
    table = tail_fit_table(tails, names)
    counts = table.Family.value_counts()
    st.write(
        " · ".join(
            f"{kind.title()}: {counts.get(kind, 0)}"
            for kind in ("exponent", "sigmoid", "plateau", "decline")
        )
    )
    st.dataframe(table.drop(columns=["Reason"]), hide_index=True)
    st.download_button(
        "Download cause trajectory fits CSV",
        table.to_csv(index=False),
        "cause_trajectory_fits.csv",
        "text/csv",
    )
    options = cause_options(frame)
    ordered_names = sorted(options)
    cause_name = st.selectbox(
        "Inspect a cause trajectory",
        ordered_names,
        index=ordered_names.index(DEMENTIA_NAME) if DEMENTIA_NAME in options else 0,
    )
    code = options[cause_name]
    fit = tails[code]
    st.info(f"{fit.status}: {fit.reason}")
    left, right = st.columns(2)
    horizon = left.slider("Trajectory display end age", 100, 300, 140, 10)
    logarithmic = right.checkbox("Logarithmic mortality scale", value=True)
    model = MortalityModel(frame, official_life_table=official, tail_fits=tails)
    settings = ProjectionConfig(
        calibrate=False,
        trend_horizon_years=0,
        pandemic=PandemicConfig(mode="hold_2021"),
        decline_residual_fraction=config.decline_residual_fraction,
    )
    ages = np.arange(70, horizon + 1)
    rates = model.rates(ages, np.full(len(ages), 2021), settings)[:, model.codes.index(code)]
    source = frame[(frame.cause_code == code) & frame.age_start.between(70, 90)]
    history = merge_dementia(raw[raw.age_start < 95])
    history = history[(history.cause_code == code) & history.year.between(2011, 2019)]
    means = history[history.age_start.between(70, 90)].groupby("age_mid").rate.mean()
    figure = go.Figure()
    figure.add_scatter(
        x=source.age_mid,
        y=source.rate,
        mode="markers",
        name="2021 closed age groups",
        marker={"size": 9, "color": "#0f766e"},
    )
    if not means.empty:
        figure.add_scatter(
            x=means.index,
            y=means.values,
            mode="lines+markers",
            name="2011–2019 mean rates",
            line={"color": "#94a3b8", "dash": "dot"},
        )
    for mask, label, dash in [
        (ages <= 94, "2021 interpolation", "solid"),
        (ages >= 94, f"Selected tail: {fit.curve.kind}", "dash"),
    ]:
        figure.add_scatter(x=ages[mask], y=rates[mask], name=label, line={"dash": dash})
    if fit.alternative_curve is not None:
        tail_ages = ages[ages >= 94]
        anchor_rate = rates[ages == 94][0]
        alternative_rates = anchor_rate * np.exp(
            fit.alternative_curve.log_multiplier(
                tail_ages - 94, decline_residual=config.decline_residual_fraction
            )
        )
        figure.add_scatter(
            x=tail_ages,
            y=alternative_rates,
            name=f"Comparison: {fit.alternative_curve.kind} ({fit.alternative_fit_ages})",
            line={"dash": "dot", "color": "#d97706"},
        )
    figure.add_vline(x=94, line_dash="dot", line_color="#94a3b8")
    figure.update_layout(
        template="plotly_white",
        height=520,
        margin={"t": 25, "b": 120},
        xaxis_title="Attained age",
        yaxis_title="Mortality rate per 100,000 per year",
        yaxis_type="log" if logarithmic else "linear",
        legend={"orientation": "h", "y": -0.25, "yanchor": "top", "x": 0},
    )
    st.plotly_chart(figure, width="stretch")
    st.caption(
        "All 95+ observations are excluded from fitting, calibration and historical trends. "
        "This chart holds 2021 levels and COVID fixed, disables calendar trends and interventions, "
        "and shows uncalibrated cause rates so they can be compared with the observations. "
        f"Declining tails retain the selected residual share: {config.decline_residual_fraction:.0%}."
    )
    with st.expander("How the first-version selection works"):
        st.markdown(
            "Positive, sufficiently large series are compared by predicting the next closed age group: "
            "70–84 → 85–89, then 70–89 → 90–94, separately for 2011–2019. "
            "Scores are mean year-wise log-RMSE, not confidence intervals. Near ties favour simpler forms. "
            "Sigmoid selection requires a material improvement and support in at least two thirds of years. "
            "A repeated decline from 85–89 to 90–94 uses that terminal branch; it has no independent "
            "older closed group for validation. Rare/zero series and COVID receive explicit plateau fallbacks. "
            "Dementia is an explicit exception: its default exponent uses 70–89, with no selection score; "
            "a sigmoid using 70–94 remains available as a comparison. "
            "No death counts or exposures are available, so these are descriptive fits. "
            "Neither sigmoid saturation levels nor centuries-long extrapolations are empirically validated."
        )
