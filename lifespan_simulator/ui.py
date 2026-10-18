"""Streamlit presentation and widget state; numerical work lives in the model modules."""

from __future__ import annotations

import json
from dataclasses import asdict
from math import inf, isfinite

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
from .data import get_display_horizon, load_official_cause_totals
from .interventions import ScenarioMap, build_scenario_map
from .mortality import MortalityModel
from .scenario_io import active_interventions_frame, parse_interventions_csv
from .simulation import project_cause_rates, project_total_rates, simulate_survival
from .trends import TrendModel, build_trend_frame

EXPLORER_CAUSE_SELECTION_KEY = "explorer_selected_causes"
INTERVENTION_CAUSES_KEY = "simulator_intervention_causes"
SLOW_START_YEAR_DEFAULT = 2040
SLOW_RATE_DEFAULT = 0.015
FAST_START_YEAR_DEFAULT = 2080
FAST_RATE_DEFAULT = 0.10
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
    if baseline == inf and scenario == inf:
        return "0.0"
    if baseline == inf:
        return None
    if scenario == inf:
        return "inf"
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
    return age_bands[band_ends >= initial_age].reset_index(drop=True)


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


def render_interventions(
    intervention_causes: list[str], all_cause_options: dict[str, str]
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
                "Floor",
            ],
            strict=True,
        ):
            column.caption(label)
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
                fast_rate = st.number_input(
                    f"Fast annual reduction for {cause_name}",
                    min_value=0.0,
                    max_value=1.0,
                    value=FAST_RATE_DEFAULT,
                    step=0.01,
                    format="%.3f",
                    key=f"fast_rate_{cause_code}",
                    label_visibility="collapsed",
                )
            with floor_col:
                floor = st.number_input(
                    f"Irreducible floor for {cause_name}",
                    min_value=1e-12,
                    max_value=1.0,
                    value=FLOOR_DEFAULT,
                    step=0.001,
                    format="%.4f",
                    key=f"floor_{cause_code}",
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
                ["kannisto", "gompertz", "flat"],
                format_func=lambda x: {
                    "kannisto": "Kannisto (logistic)",
                    "gompertz": "Gompertz (exponential)",
                    "flat": "Flat 95+ (comparison only)",
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
            pandemic=PandemicConfig(pandemic_mode, half_life, residual),
            trend_floor=trend_floor,
            trend_ceiling=trend_ceiling,
            trend_horizon_years=trend_horizon if use_trends else 0,
            max_age=max_age,
        ),
    )


def render_simulator_tab(frame: pd.DataFrame, official: pd.DataFrame, trend: TrendModel) -> None:
    initial_age, use_trends, config = render_projection_controls()
    model = MortalityModel(frame, trend.slope_table if use_trends else None, official)
    if use_trends:
        st.caption(
            f"Using {trend.significant_count} of {trend.total_count} non-COVID cause-age trends fitted through 2019. "
            "The baseline is conditional on survival to the selected starting age and includes the chosen COVID path."
        )
    options = cause_options(frame)
    with st.expander("Import intervention settings"):
        uploaded = st.file_uploader("Load active interventions CSV", type=["csv"])
        if st.button("Load interventions", disabled=uploaded is None):
            try:
                names, inputs = parse_interventions_csv(uploaded, options)
                apply_loaded_interventions(names, inputs, options)
                st.success(f"Loaded {len(names)} intervention(s).")
            except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as error:
                st.error(f"Could not load interventions CSV. {error}")
    selected = st.multiselect(
        "Causes to modify in the scenario", sorted(options), key=INTERVENTION_CAUSES_KEY
    )
    scenarios = render_interventions(selected, options)
    baseline = simulate_survival(model, initial_age, config=config)
    scenario = simulate_survival(model, initial_age, scenarios, config) if scenarios else baseline

    metrics = st.columns(4)
    metrics[0].metric("Baseline mean age at death", format_metric(baseline.life_expectancy_age))
    metrics[1].metric(
        "Scenario mean age at death",
        format_metric(scenario.life_expectancy_age),
        format_delta(baseline.life_expectancy_age, scenario.life_expectancy_age),
        help="Unresolved means the remaining contribution to the mean cannot be bounded within the numerical error limit. See the mean's bounds below; the simulation has still completed.",
    )
    metrics[2].metric("Scenario remaining years", format_metric(scenario.remaining_life_expectancy))
    metrics[3].metric("Scenario median age", format_percentile_age(scenario.percentiles["p50"]))
    if not baseline.converged or not scenario.converged:
        st.warning(
            f"Simulation completed, but the full mean is not resolved by age {config.max_age}. "
            "Survivors beyond that age could still materially affect the mean under the selected assumptions. "
            "The median can still be known: it only requires half of the cohort to have died. "
            "See the survival percentages and numerical bounds below. Increasing the horizon may narrow "
            "the bounds, but does not guarantee a resolved mean."
        )
    diagnostics = []
    for label, result in [("Baseline", baseline), ("Scenario", scenario)]:
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
                "Full mean resolved": result.converged,
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
            "A resolved full mean has at most "
            f"{config.expectation_tolerance:g} years of omitted tail contribution under the selected assumptions. "
            "These are numerical bounds within the model, not confidence intervals or bounds on forecast uncertainty."
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
    horizon = max(get_display_horizon(initial_age), band_end)
    all_rates = project_cause_rates(
        model, initial_age, list(model.codes), scenarios, horizon, config
    )
    ranked = (
        all_rates[all_rates.age.between(max(initial_age, band_start), band_end)]
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
        "historical_trends": "2011–2019" if use_trends else "disabled",
        "reference_population": "Spain, males, 2021",
    }
    st.download_button(
        "Download model assumptions JSON",
        json.dumps(assumptions, indent=2),
        "assumptions.json",
        "application/json",
    )
    render_sensitivity(model, initial_age, scenarios, config)


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
                x=results[column],
                name=column,
                orientation="h",
                marker_color=color,
            )
        )
    figure.update_layout(
        barmode="group",
        height=max(480, len(results) * 34),
        xaxis_title="Mean age at death (resolved projections only)",
        yaxis={"autorange": "reversed"},
        template="plotly_white",
    )
    st.plotly_chart(figure, width="stretch")
    st.dataframe(results, hide_index=True)
    st.caption(
        f"Blank means/percentiles extend beyond the age-{config.max_age} calculation or fail its remainder bound. "
        "Restricted expected years use the same horizon in every row. Slope variations have no effect in flat mode; "
        "trend variations have no effect when trends are disabled."
    )
    st.download_button(
        "Download sensitivity CSV", results.to_csv(index=False), "sensitivity.csv", "text/csv"
    )


def render_validation_tab(frame: pd.DataFrame, official: pd.DataFrame) -> None:
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
    expectancy = compare_life_expectancy(frame, official)
    st.markdown("**Remaining life expectancy at selected exact ages**")
    st.dataframe(
        expectancy.rename(
            columns={
                "age": "Starting age",
                "official_remaining_years": "Official years",
                "uncalibrated_flat": "Source + flat 95+",
                "uncalibrated_kannisto": "Source + Kannisto",
                "calibrated_kannisto": "Calibrated + Kannisto",
                "calibrated_difference": "Difference (years)",
            }
        ),
        hide_index=True,
    )
    st.caption(
        "Official life expectancy is published to 0.1 years. Mx is interpreted as a constant hazard within each year. "
        "The 95+ rate is an open-group average; Kannisto is fitted to ages 80–94 and extrapolated beyond 94."
    )
    fit = MortalityModel(frame, official_life_table=official).old_age_fit(ProjectionConfig())
    with st.expander("Old-age fit and limitations"):
        st.json(fit)
        st.markdown(
            "The fit uses least squares on logit mortality, an asymptote of 1/year, and an anchor at age 94. "
            "It is inspired by the [HMD methodology](https://www.mortality.org/File/GetDocument/Public/Docs/MethodsProtocolV6.pdf), "
            "but does not reproduce its exposure-weighted estimation. Cause shares are fixed after 95. "
            "Interventions can reduce the old-age hazard, so extreme scenarios can still have very long lifespans."
        )
    st.download_button(
        "Download life-table comparison CSV",
        expectancy.to_csv(index=False),
        "reference_comparison.csv",
        "text/csv",
    )
