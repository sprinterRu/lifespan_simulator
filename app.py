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
    get_display_horizon,
    get_simulator_frame,
    load_mortality_data,
)
from lifespan_simulator.simulation import build_scenario_map, project_cause_rates, project_total_rates, simulate_survival
from lifespan_simulator.trends import build_trend_frame

st.set_page_config(page_title="Lifespan Simulator", layout="wide")

EXPLORER_CAUSE_SELECTION_KEY = "explorer_selected_causes"
INTERVENTION_CAUSES_KEY = "simulator_intervention_causes"
ACTIVE_INTERVENTION_COLUMNS = ("Cause", "Effective year", "New rate")


@st.cache_data(show_spinner=False)
def load_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the source mortality data and derive the simulator subset.

    Returns:
        A pair of DataFrames. The first is the complete normalized mortality
        dataset returned by ``load_mortality_data(DATA_PATH)``. The second is
        the simulator-ready 2021 subset returned by ``get_simulator_frame`` and
        used by the survival model.
    """
    raw = load_mortality_data(DATA_PATH)
    return raw, get_simulator_frame(raw)


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
    """Build the display-name-to-code lookup used by simulator controls.

    Args:
        frame: Mortality DataFrame currently being used by the simulator tab.
            In normal app execution this is ``simulator_frame`` from
            :func:`load_frames`, so it contains one row per 2021 age bucket and
            non-overlapping cause code.

    Returns:
        Dictionary keyed by human-readable cause name with ICD-10 cause code as
        the value. The multiselect displays the keys, while simulation functions
        consume the values.
    """
    return frame.drop_duplicates("cause_name").set_index("cause_name")["cause_code"].to_dict()


def cause_names_by_code(frame: pd.DataFrame) -> dict[str, str]:
    """Build the code-to-display-name lookup used by tables.

    Args:
        frame: Mortality DataFrame currently being used by the simulator tab.
            In normal app execution this is ``simulator_frame`` from
            :func:`load_frames`.

    Returns:
        Dictionary keyed by ICD-10 cause code with the human-readable cause name
        as the value. The Active interventions table uses this to display cause
        names for the scenario map, which is keyed by cause code.
    """
    return frame.drop_duplicates("cause_code").set_index("cause_code")["cause_name"].to_dict()


def active_interventions_frame(
    scenario_map: dict[str, dict[str, float | int]],
    cause_name_lookup: dict[str, str],
) -> pd.DataFrame:
    """Convert active scenario settings into the CSV-compatible table shape.

    Args:
        scenario_map: Normalized intervention map returned by
            ``build_scenario_map``. It is keyed by ICD-10 cause code and stores
            ``effective_year`` plus replacement ``rate`` values from the
            simulator controls or a loaded CSV.
        cause_name_lookup: Code-to-name dictionary returned by
            :func:`cause_names_by_code`. It comes from the same
            ``simulator_frame`` as the scenario controls, ensuring table labels
            match the selectable causes.

    Returns:
        DataFrame with exactly the columns in
        :data:`ACTIVE_INTERVENTION_COLUMNS`: ``Cause``, ``Effective year``, and
        ``New rate``. This is the same shape expected when loading a saved CSV.
    """
    return pd.DataFrame(
        [
            {
                "Cause": cause_name_lookup[cause_code],
                "Effective year": config["effective_year"],
                "New rate": config["rate"],
            }
            for cause_code, config in scenario_map.items()
        ],
        columns=ACTIVE_INTERVENTION_COLUMNS,
    )


def parse_interventions_csv(
    uploaded_file,
    all_cause_options: dict[str, str],
) -> tuple[list[str], dict[str, dict[str, float | int]]]:
    """Read and validate an uploaded Active interventions CSV file.

    Args:
        uploaded_file: File-like object returned by ``st.file_uploader`` in
            ``render_simulator_tab``. Tests may pass any seekable file-like
            object, such as ``io.StringIO``, with the same CSV text.
        all_cause_options: Display-name-to-code dictionary returned by
            :func:`cause_options` from the current ``simulator_frame``. It is
            used to verify that every ``Cause`` value in the CSV exists in the
            simulator's non-overlapping 2021 cause set.

    Returns:
        A pair of values. The first is the ordered list of cause display names
        from the CSV; it is used to seed the Streamlit multiselect. The second
        is a raw scenario-input mapping keyed by cause code with
        ``effective_year`` and ``rate`` values; it has the same shape as the
        dictionary built from manual Streamlit inputs before calling
        ``build_scenario_map``.

    Raises:
        ValueError: If required columns are missing, the file contains no
            intervention rows, a cause is unknown or duplicated, the effective
            year is not an integer in the supported 2021-2120 range, or the
            replacement rate is negative or not numeric.
        pandas.errors.EmptyDataError: If pandas cannot find any CSV columns.
        pandas.errors.ParserError: If pandas cannot parse the uploaded CSV.
    """
    uploaded_file.seek(0)
    frame = pd.read_csv(uploaded_file)
    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame.dropna(how="all")

    missing_columns = [column for column in ACTIVE_INTERVENTION_COLUMNS if column not in frame.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required column(s): {missing}.")

    if frame.empty:
        raise ValueError("The uploaded CSV does not contain any interventions.")

    cause_names: list[str] = []
    loaded_inputs: dict[str, dict[str, float | int]] = {}

    for row_number, row in enumerate(frame.to_dict(orient="records"), start=2):
        cause_name = str(row["Cause"]).strip()
        if cause_name not in all_cause_options:
            raise ValueError(f"Unknown cause on row {row_number}: {cause_name}.")
        if cause_name in cause_names:
            raise ValueError(f"Duplicate cause on row {row_number}: {cause_name}.")

        effective_year = pd.to_numeric(row["Effective year"], errors="coerce")
        if pd.isna(effective_year) or int(effective_year) != effective_year:
            raise ValueError(f"Effective year must be a whole number on row {row_number}.")
        effective_year = int(effective_year)
        if not 2021 <= effective_year <= 2120:
            raise ValueError(f"Effective year must be between 2021 and 2120 on row {row_number}.")

        new_rate = pd.to_numeric(row["New rate"], errors="coerce")
        if pd.isna(new_rate) or float(new_rate) < 0:
            raise ValueError(f"New rate must be a non-negative number on row {row_number}.")

        cause_names.append(cause_name)
        loaded_inputs[all_cause_options[cause_name]] = {
            "effective_year": effective_year,
            "rate": float(new_rate),
        }

    return cause_names, loaded_inputs


def apply_loaded_interventions(
    cause_names: list[str],
    scenario_inputs: dict[str, dict[str, float | int]],
    all_cause_options: dict[str, str],
) -> None:
    """Seed Streamlit widget state from a validated intervention CSV.

    Args:
        cause_names: Ordered cause display names returned by
            :func:`parse_interventions_csv`. They become the selected values
            for the ``Causes to modify in the scenario`` multiselect.
        scenario_inputs: Raw scenario-input mapping returned by
            :func:`parse_interventions_csv`. It is keyed by cause code and
            contains ``effective_year`` plus ``rate`` values to copy into the
            per-cause number-input widgets.
        all_cause_options: Display-name-to-code dictionary returned by
            :func:`cause_options`. It is used to translate each display name in
            ``cause_names`` into the cause code embedded in the widget keys.
    """
    st.session_state[INTERVENTION_CAUSES_KEY] = cause_names
    for cause_name in cause_names:
        cause_code = all_cause_options[cause_name]
        config = scenario_inputs[cause_code]
        st.session_state[f"year_{cause_code}"] = int(config["effective_year"])
        st.session_state[f"rate_{cause_code}"] = float(config["rate"])


def minimum_rates_by_code(frame: pd.DataFrame) -> dict[str, float]:
    """Find the lowest baseline rate available for each simulator cause.

    Args:
        frame: Mortality DataFrame currently being used by the simulator tab.
            In normal app execution this is ``simulator_frame`` from
            :func:`load_frames`, filtered to the 2021 non-overlapping cause set.

    Returns:
        Dictionary keyed by ICD-10 cause code. Each value is the minimum
        mortality rate per 100,000 observed for that cause across the simulator
        age buckets. The simulator uses this as the default replacement rate
        when a user first selects an intervention cause.
    """
    return frame.groupby("cause_code")["rate"].min().astype(float).to_dict()


def default_explorer_causes(raw_frame: pd.DataFrame, age_label: str) -> list[str]:
    """Choose initial cause selections for the historical explorer.

    Args:
        raw_frame: Complete normalized mortality dataset returned as the first
            value from :func:`load_frames`. Unlike ``simulator_frame``, it still
            includes aggregate categories because the explorer does not sum
            causes together.
        age_label: Human-readable age bucket label selected in the explorer
            ``Age group`` selectbox, such as ``35-39`` or ``95+``.

    Returns:
        Up to three cause names with the highest 2021 mortality rates in the
        selected age group. These names seed the explorer multiselect the first
        time the tab renders.
    """
    return (
        raw_frame[(raw_frame["age_label"] == age_label) & (raw_frame["year"] == 2021)]
        .sort_values("rate", ascending=False)
        .head(3)["cause_name"]
        .tolist()
    )


def format_delta(baseline: float, scenario: float) -> str:
    """Format the difference between a scenario metric and its baseline value.

    Args:
        baseline: Baseline numeric metric produced by a baseline
            :class:`lifespan_simulator.simulation.SimulationResult`, such as
            remaining life expectancy.
        scenario: Scenario numeric metric produced by the corresponding
            scenario :class:`lifespan_simulator.simulation.SimulationResult`.

    Returns:
        Display string for Streamlit metric deltas. Finite values are formatted
        with an explicit sign and one decimal place; infinite or unavailable
        combinations are converted to stable labels.
    """
    if baseline == inf and scenario == inf:
        return "0.0"
    if baseline == inf:
        return "n/a"
    if scenario == inf:
        return "inf"
    return f"{scenario - baseline:+.1f}"


def format_optional_delta(baseline: float | None, scenario: float | None) -> str:
    """Format a metric delta when either side may be unavailable.

    Args:
        baseline: Baseline value from the simulation result. Percentile values
            can be ``None`` when a zero-hazard tail never reaches the requested
            death percentile.
        scenario: Scenario value from the simulation result. It follows the
            same ``None`` convention as ``baseline``.

    Returns:
        ``"n/a"`` if either input is ``None``; otherwise the formatted delta
        returned by :func:`format_delta`.
    """
    if baseline is None or scenario is None:
        return "n/a"
    return format_delta(baseline, scenario)


def format_slope_ci(stats: dict[str, object]) -> str:
    """Format the 95% confidence interval for an explorer trend slope.

    Args:
        stats: Statistics mapping returned by
            ``lifespan_simulator.trends.build_trend_frame`` for one selected
            cause and age group. It must include ``slope_ci_lower`` and
            ``slope_ci_upper`` values when enough yearly observations are
            available to fit the regression.

    Returns:
        Display string in ``[lower, upper]`` form with signed bounds and two
        decimals. Returns ``"n/a"`` when either bound is missing or NaN.
    """
    lower = stats.get("slope_ci_lower")
    upper = stats.get("slope_ci_upper")
    if pd.isna(lower) or pd.isna(upper):
        return "n/a"
    return f"[{float(lower):+.2f}, {float(upper):+.2f}]"


def format_slope_significance(stats: dict[str, object]) -> str:
    """Describe whether the slope confidence interval excludes zero.

    Args:
        stats: Statistics mapping returned by
            ``lifespan_simulator.trends.build_trend_frame`` for one selected
            cause and age group. It provides ``slope_significant`` after the
            trend utility compares the slope's 95% confidence interval to zero.

    Returns:
        ``"Yes"`` when the slope's 95% confidence interval excludes zero,
        ``"No"`` when the interval includes zero, and ``"n/a"`` when the slope
        interval could not be estimated.
    """
    lower = stats.get("slope_ci_lower")
    upper = stats.get("slope_ci_upper")
    if pd.isna(lower) or pd.isna(upper):
        return "n/a"
    return "Yes" if bool(stats.get("slope_significant")) else "No"


def available_top_cause_age_bands(simulator_frame: pd.DataFrame, initial_age: int) -> pd.DataFrame:
    """Filter age bands that can be used to rank top scenario causes.

    Args:
        simulator_frame: Simulator-ready mortality DataFrame returned by
            :func:`load_frames`. It provides the available age bucket starts,
            ends, and labels.
        initial_age: Cohort starting age selected by the ``Initial age`` slider
            in ``render_simulator_tab``. Age bands ending before this age are
            excluded because they are not reached by the modeled cohort.

    Returns:
        DataFrame with ``age_start``, ``age_end``, and ``age_label`` columns for
        every age bucket that overlaps or follows ``initial_age``. The simulator
        uses it to populate the ``Rank top causes by age group`` selectbox.
    """
    age_bands = (
        simulator_frame.drop_duplicates("age_start")
        .sort_values("age_start")[["age_start", "age_end", "age_label"]]
        .reset_index(drop=True)
    )
    band_ends = age_bands["age_end"].fillna(age_bands["age_start"])
    return age_bands[band_ends >= initial_age].reset_index(drop=True)


def default_top_cause_age_start(age_starts: list[int]) -> int:
    """Pick the default age bucket for ranking top causes.

    Args:
        age_starts: Ordered age-bucket starts extracted from
            :func:`available_top_cause_age_bands` for the current initial age.

    Returns:
        ``85`` when that age bucket is available, because older-age mortality
        tends to be most relevant to the life expectancy display. Otherwise the
        first available bucket is returned.
    """
    return 85 if 85 in age_starts else age_starts[0]


def age_band_details(simulator_frame: pd.DataFrame, age_start: int) -> tuple[int, int, str]:
    """Return numeric bounds and label for one simulator age bucket.

    Args:
        simulator_frame: Simulator-ready mortality DataFrame returned by
            :func:`load_frames`. It contains repeated rows for each age bucket,
            one per cause.
        age_start: Age-bucket start selected by the
            ``Rank top causes by age group`` selectbox.

    Returns:
        Tuple of ``(age_start, age_end, age_label)``. For the open-ended
        ``95+`` bucket, ``age_end`` is set equal to ``age_start`` so callers can
        use ordinary inclusive numeric comparisons.
    """
    age_band = simulator_frame.loc[simulator_frame["age_start"] == age_start].iloc[0]
    age_end = age_start if pd.isna(age_band["age_end"]) else int(age_band["age_end"])
    return age_start, age_end, str(age_band["age_label"])


def scenario_top_causes(
    simulator_frame: pd.DataFrame,
    initial_age: int,
    age_band_start: int,
    scenario_map: dict[str, dict[str, float | int]],
    limit: int = 5,
) -> pd.DataFrame:
    """Rank causes by average scenario mortality within a chosen age band.

    Args:
        simulator_frame: Simulator-ready mortality DataFrame returned by
            :func:`load_frames`. It supplies the full non-overlapping cause set
            and baseline rates used for projection.
        initial_age: Cohort starting age selected by the simulator slider. It
            is passed through to ``project_cause_rates`` so calendar years line
            up with the scenario.
        age_band_start: Start age selected in the ``Rank top causes by age
            group`` selectbox. It determines the age interval used for ranking.
        scenario_map: Normalized intervention map returned by
            ``build_scenario_map`` from the current manual or loaded scenario
            controls.
        limit: Maximum number of causes to return. The app uses the default of
            five to choose the automatic chart lines.

    Returns:
        DataFrame with the highest-ranking ``cause_code`` and ``cause_name``
        rows plus their mean ``scenario_rate`` over the selected band. The
        returned frame also stores the age-band label in ``attrs["age_label"]``
        for use in the chart caption.
    """
    cause_codes = simulator_frame.drop_duplicates("cause_code")["cause_code"].tolist()
    band_start, band_end, age_label = age_band_details(simulator_frame, age_band_start)
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
    return ranked


def render_intro(simulator_frame: pd.DataFrame) -> None:
    """Render the app title, caption, and assumptions expander.

    Args:
        simulator_frame: Simulator-ready mortality DataFrame returned by
            :func:`load_frames`. The assumptions text reads it to display the
            number of mutually exclusive cause series included in the simulator.
    """
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
    """Render the simulator workflow and all scenario-dependent outputs.

    Args:
        simulator_frame: Simulator-ready mortality DataFrame returned by
            :func:`load_frames`. It comes from the 2021 source mortality data
            after excluding overlapping roll-up causes. This function uses it
            to populate controls, parse loaded interventions, build scenario
            maps, run baseline and scenario simulations, and draw the simulator
            charts and tables.
    """
    all_cause_options = cause_options(simulator_frame)
    cause_name_lookup = cause_names_by_code(simulator_frame)
    cause_minimum_rates = minimum_rates_by_code(simulator_frame)
    age_min = int(simulator_frame["age_start"].min())
    age_max = int(simulator_frame["age_start"].max())

    controls_left, controls_middle, controls_right = st.columns([0.8, 1, 1.8])
    with controls_left:
        initial_age = st.slider("Initial age", min_value=age_min, max_value=age_max, value=35, step=1)
    top_cause_age_bands = available_top_cause_age_bands(simulator_frame, initial_age)
    top_cause_age_starts = top_cause_age_bands["age_start"].astype(int).tolist()
    top_cause_age_label_lookup = top_cause_age_bands.set_index("age_start")["age_label"].to_dict()
    default_age_start = default_top_cause_age_start(top_cause_age_starts)
    if st.session_state.get("top_cause_age_start") not in top_cause_age_starts:
        st.session_state["top_cause_age_start"] = default_age_start
    with controls_middle:
        top_cause_age_start = st.selectbox(
            "Rank top causes by age group",
            options=top_cause_age_starts,
            format_func=lambda age_start: top_cause_age_label_lookup[age_start],
            help="Choose the attained-age band used to rank the automatic top-5 cause lines.",
            key="top_cause_age_start",
        )
    with controls_right:
        uploaded_interventions = st.file_uploader(
            "Load active interventions CSV",
            type=["csv"],
        )
        if st.button("Load interventions", disabled=uploaded_interventions is None):
            try:
                loaded_cause_names, loaded_inputs = parse_interventions_csv(
                    uploaded_interventions,
                    all_cause_options,
                )
                apply_loaded_interventions(loaded_cause_names, loaded_inputs, all_cause_options)
                st.success(f"Loaded {len(loaded_cause_names)} intervention(s).")
            except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as error:
                st.error(f"Could not load interventions CSV. {error}")

        intervention_causes = st.multiselect(
            "Causes to modify in the scenario",
            options=sorted(all_cause_options.keys()),
            help=(
                "Set a future calendar year and replacement rate for each selected cause. "
                "The replacement rate defaults to the cause's lowest 2021 rate across age groups."
            ),
            key=INTERVENTION_CAUSES_KEY,
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
                    value=cause_minimum_rates[cause_code],
                    step=0.01,
                    format="%.2f",
                    key=f"rate_{cause_code}",
                    label_visibility="collapsed",
                )
            scenario_inputs[cause_code] = {"effective_year": effective_year, "rate": replacement_rate}

    scenario_map = build_scenario_map(scenario_inputs)
    baseline_result = simulate_survival(simulator_frame, initial_age)
    scenario_result = simulate_survival(simulator_frame, initial_age, scenario_map)
    top_scenario_causes = scenario_top_causes(
        simulator_frame,
        initial_age,
        top_cause_age_start,
        scenario_map,
        limit=5,
    )
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
            st.caption(
                f"The chart shows the top 5 causes in the current scenario ranked using "
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
                active_interventions_frame(scenario_map, cause_name_lookup),
                hide_index=True,
                use_container_width=True,
            )


def render_explorer_tab(raw_frame: pd.DataFrame) -> None:
    """Render the historical mortality-rate trend explorer.

    Args:
        raw_frame: Complete normalized mortality dataset returned as the first
            value from :func:`load_frames`. It includes all source causes and
            all years from the CSV because the explorer visualizes individual
            cause histories rather than summing rates for survival simulation.
    """
    age_labels = raw_frame.drop_duplicates("age_start").sort_values("age_start")["age_label"].tolist()
    all_causes = raw_frame.drop_duplicates("cause_name").sort_values("cause_name")
    all_cause_names = all_causes["cause_name"].tolist()
    name_to_code = all_causes.set_index("cause_name")["cause_code"].to_dict()

    control_col, toggle_col = st.columns([1.4, 0.6])
    with control_col:
        age_label = st.selectbox("Age group", options=age_labels, index=0)
        # Seed the explorer with default causes only once; otherwise Streamlit's
        # rerun after changing age group would replace the user's selection.
        if EXPLORER_CAUSE_SELECTION_KEY not in st.session_state:
            st.session_state[EXPLORER_CAUSE_SELECTION_KEY] = default_explorer_causes(raw_frame, age_label)
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
        subset = raw_frame[(raw_frame["age_label"] == age_label) & (raw_frame["cause_code"] == cause_code)].copy()
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

    st.plotly_chart(build_trend_figure(trend_frames, age_label, log_scale), use_container_width=True)
    st.caption(
        "Chart bands show the 95% interval around the fitted mean trend line, not a forecast interval. "
        "A slope is marked as excluding 0 when its 95% confidence interval does not include zero."
    )
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)


def main() -> None:
    """Run the Streamlit app from initial data load through tab rendering."""
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
