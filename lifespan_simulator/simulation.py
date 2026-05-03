"""Piecewise-constant mortality simulation utilities.

The functions in this module receive the normalized 2021 simulator dataset
prepared in :mod:`lifespan_simulator.data` and optional intervention settings
collected by the Streamlit UI. They convert cause-specific mortality rates into
annual hazards, apply any selected future replacement rates, and return the
tables needed by the app's charts and metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, inf, log
from typing import Any

import pandas as pd

from .data import get_age_bucket_start

START_YEAR = 2021
# Mortality rates in the source data are expressed per 100,000 population.
# Dividing by this value converts a rate into the annual hazard used by the
# exponential survival model.
RATE_SCALE = 100_000.0
# Labels are death-distribution percentiles. For example, p75 is the age by
# which 75% of the synthetic cohort has died.
PERCENTILES = {
    "p50": 0.50,
    "p75": 0.75,
    "p90": 0.90,
    "p95": 0.95,
    "p99": 0.99,
}
TrendTable = dict[int, dict[str, float]]


@dataclass(frozen=True)
class SimulationResult:
    """Full result for one synthetic cohort simulation.

    Instances are created by :func:`simulate_survival` after it has projected
    the annual hazards for a particular initial age and scenario. The Streamlit
    app reads these attributes directly when rendering the headline metrics,
    survival chart, and percentile table.

    Attributes:
        annual_projection: DataFrame built inside :func:`simulate_survival`.
            Each row represents one modeled one-year age interval before the
            open-ended terminal tail and contains age bounds, calendar year,
            total mortality rate, hazard, and survival at the start and end of
            the interval.
        survival_curve: Plot-ready DataFrame built by
            :func:`_build_survival_curve`. It contains ``age`` and ``survival``
            columns and extends into the terminal tail far enough to show the
            modeled percentile ages.
        tail_start_age: First attained age handled by the terminal tail. It is
            computed by :func:`simulate_survival` from age 95 and any later
            intervention effective years, because the source data's final age
            bucket is ``95+``.
        tail_hazard: Constant annual hazard assumed from ``tail_start_age``
            onward. It comes from the total resolved mortality rate at the tail
            boundary divided by :data:`RATE_SCALE`.
        survival_at_tail: Cohort survival probability at ``tail_start_age``.
            This is the running survival value after all explicit annual rows
            have been processed.
        remaining_life_expectancy: Expected remaining years of life from the
            cohort's initial age under the scenario passed to
            :func:`simulate_survival`.
        life_expectancy_age: Attained age equal to initial age plus
            ``remaining_life_expectancy``. This is the age shown in the app as
            total life expectancy for the selected starting age.
        percentiles: Mapping from labels in :data:`PERCENTILES` to modeled
            attained ages. Values are ``None`` only when a zero-hazard tail
            means the target percentile is never reached.
    """

    annual_projection: pd.DataFrame
    survival_curve: pd.DataFrame
    tail_start_age: int
    tail_hazard: float
    survival_at_tail: float
    remaining_life_expectancy: float
    life_expectancy_age: float
    percentiles: dict[str, float | None]


def _build_rate_table(frame: pd.DataFrame) -> tuple[dict[int, dict[str, float]], dict[str, str]]:
    """Convert the simulator mortality frame into fast lookup dictionaries.

    Args:
        frame: Normalized simulator mortality DataFrame. In the app this is
            ``simulator_frame`` returned by
            ``get_simulator_frame(load_mortality_data(DATA_PATH))`` in
            ``app.load_frames``. It must contain one 2021 row per age bucket
            and cause, with at least ``age_start``, ``cause_code``,
            ``cause_name``, and ``rate`` columns.

    Returns:
        A pair of dictionaries. The first maps an age-bucket start, such as
        ``35`` or ``95``, to a nested mapping from ICD-10 cause code to the
        source mortality rate for that bucket. The second maps each cause code
        to the human-readable cause name used in tables and charts.
    """
    rate_table = (
        frame.pivot_table(index="age_start", columns="cause_code", values="rate", aggfunc="first")
        .fillna(0.0)
        .to_dict(orient="index")
    )
    cause_names = frame.drop_duplicates("cause_code").set_index("cause_code")["cause_name"].to_dict()
    return rate_table, cause_names


def build_scenario_map(raw_scenarios: dict[str, dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    """Normalize raw intervention inputs into the simulation scenario format.

    Args:
        raw_scenarios: Mapping keyed by cause code. In the app it is assembled
            in ``render_simulator_tab`` from the selected intervention causes
            and the Streamlit number inputs named ``year_<cause_code>`` and
            ``rate_<cause_code>``. Each value must provide ``effective_year``,
            the first calendar year when the intervention applies, and ``rate``,
            the replacement mortality rate per 100,000 population.

    Returns:
        A clean scenario map keyed by cause code. ``effective_year`` is coerced
        to ``int`` and ``rate`` is coerced to ``float`` and clamped to zero or
        above so downstream projection code can use the values without knowing
        about Streamlit widget types.
    """
    scenario_map: dict[str, dict[str, float | int]] = {}
    for cause_code, config in raw_scenarios.items():
        scenario_map[cause_code] = {
            "effective_year": int(config["effective_year"]),
            "rate": max(0.0, float(config["rate"])),
        }
    return scenario_map


def _resolve_rate(
    age: int,
    calendar_year: int,
    cause_code: str,
    rate_table: dict[int, dict[str, float]],
    scenario_map: dict[str, dict[str, float | int]],
    trend_table: TrendTable | None = None,
) -> float:
    """Return the mortality rate that applies for one cause at one age/year.

    Args:
        age: Attained age being projected. Callers pass loop values starting at
            the user's selected initial age and advancing one year at a time.
        calendar_year: Calendar year corresponding to ``age``. Projection
            functions compute it as ``start_year + (age - initial_age)``.
        cause_code: ICD-10 cause code being resolved. It comes from the
            simulator dataset, from selected chart causes, or from the keys of
            a scenario map.
        rate_table: Baseline rate lookup returned by :func:`_build_rate_table`.
            It provides source 2021 mortality rates by age-bucket start and
            cause code.
        scenario_map: Normalized intervention map returned by
            :func:`build_scenario_map`. If it contains ``cause_code`` and
            ``calendar_year`` is greater than or equal to that intervention's
            ``effective_year``, the scenario replacement rate is used before
            any historical trend adjustment.
        trend_table: Optional nested lookup returned by
            ``lifespan_simulator.trends.build_trend_model``. When present, it
            supplies significant linear annual slopes by age bucket and cause
            code. Non-significant cause-age trends are represented as ``0.0``.

    Returns:
        The applicable mortality rate per 100,000 population. Interventions
        override all other behavior. Otherwise, if a trend table is supplied,
        the 2021 baseline rate is adjusted by ``slope * (calendar_year - 2021)``
        and floored at zero. Without a trend table, the fixed 2021 baseline
        rate is returned.
    """
    bucket_start = get_age_bucket_start(age)
    baseline_rate = float(rate_table[bucket_start].get(cause_code, 0.0))
    scenario = scenario_map.get(cause_code)
    if scenario and calendar_year >= int(scenario["effective_year"]):
        return float(scenario["rate"])
    if trend_table is not None:
        slope = float(trend_table.get(bucket_start, {}).get(cause_code, 0.0))
        return max(0.0, baseline_rate + slope * (calendar_year - START_YEAR))
    return baseline_rate


def project_cause_rates(
    frame: pd.DataFrame,
    initial_age: int,
    selected_causes: list[str],
    scenario_map: dict[str, dict[str, float | int]],
    max_age: int,
    start_year: int = START_YEAR,
    trend_table: TrendTable | None = None,
) -> pd.DataFrame:
    """Project baseline and scenario rates by attained age for selected causes.

    Args:
        frame: Normalized simulator mortality DataFrame from
            ``get_simulator_frame``. It supplies the 2021 baseline rate for
            each age bucket and cause code.
        initial_age: Cohort starting age selected by the user with the
            ``Initial age`` slider in ``render_simulator_tab``.
        selected_causes: Cause codes to include in the chart-ready output. In
            the app this is the union of the automatically ranked top scenario
            causes and the causes explicitly selected for intervention.
        scenario_map: Normalized intervention map returned by
            :func:`build_scenario_map`. It may be empty for the baseline case.
        max_age: Last attained age to include. The app passes the display
            horizon returned by ``get_display_horizon(initial_age)`` or the end
            of the age band being used for top-cause ranking.
        start_year: Calendar year assigned to ``initial_age``. The default is
            :data:`START_YEAR`, matching the 2021 baseline mortality schedule.
        trend_table: Optional significant-slope lookup from
            ``build_trend_model``. When present, both the no-intervention
            baseline projection and scenario projection include historical
            trend adjustments before scenario interventions are applied.

    Returns:
        DataFrame with one row per projected age and selected cause. Columns
        include ``age``, ``calendar_year``, ``cause_code``, ``cause_name``,
        ``baseline_rate``, and ``scenario_rate``. ``baseline_rate`` means
        "without active interventions"; it can still include trend adjustments
        when ``trend_table`` is supplied. Chart builders consume this frame
        directly.
    """
    rate_table, cause_names = _build_rate_table(frame)
    rows: list[dict[str, Any]] = []

    for age in range(initial_age, max_age + 1):
        calendar_year = start_year + (age - initial_age)
        for cause_code in selected_causes:
            rows.append(
                {
                    "age": age,
                    "calendar_year": calendar_year,
                    "cause_code": cause_code,
                    "cause_name": cause_names[cause_code],
                    "baseline_rate": _resolve_rate(
                        age,
                        calendar_year,
                        cause_code,
                        rate_table,
                        {},
                        trend_table,
                    ),
                    "scenario_rate": _resolve_rate(
                        age,
                        calendar_year,
                        cause_code,
                        rate_table,
                        scenario_map,
                        trend_table,
                    ),
                }
            )

    return pd.DataFrame(rows)


def project_total_rates(
    frame: pd.DataFrame,
    initial_age: int,
    scenario_map: dict[str, dict[str, float | int]],
    max_age: int,
    start_year: int = START_YEAR,
    trend_table: TrendTable | None = None,
) -> pd.DataFrame:
    """Project total baseline and scenario mortality rates by attained age.

    Args:
        frame: Normalized simulator mortality DataFrame from
            ``get_simulator_frame``. It is already filtered to the
            non-overlapping cause set so summing rates does not double-count
            aggregate roll-up categories.
        initial_age: Cohort starting age selected by the Streamlit
            ``Initial age`` slider.
        scenario_map: Normalized intervention map returned by
            :func:`build_scenario_map`. Each matching cause can replace the
            baseline rate from its effective year onward.
        max_age: Last attained age to project. The app typically passes the
            display horizon returned by ``get_display_horizon(initial_age)``.
        start_year: Calendar year assigned to ``initial_age``. It defaults to
            :data:`START_YEAR`, the same year used to filter the simulator
            mortality frame.
        trend_table: Optional significant-slope lookup from
            ``build_trend_model``. When present, both total series include
            historical trend adjustments, and the scenario series additionally
            applies active interventions.

    Returns:
        DataFrame with one row per projected attained age. It contains
        ``age``, ``calendar_year``, ``baseline_rate`` summed across all
        simulator causes without interventions, and ``scenario_rate`` after
        applying any active interventions. Either series can include trend
        adjustments when ``trend_table`` is supplied.
    """
    rate_table, cause_names = _build_rate_table(frame)
    cause_codes = sorted(cause_names)
    rows: list[dict[str, Any]] = []

    for age in range(initial_age, max_age + 1):
        calendar_year = start_year + (age - initial_age)
        baseline_total = sum(
            _resolve_rate(age, calendar_year, code, rate_table, {}, trend_table) for code in cause_codes
        )
        scenario_total = sum(
            _resolve_rate(age, calendar_year, code, rate_table, scenario_map, trend_table) for code in cause_codes
        )
        rows.append(
            {
                "age": age,
                "calendar_year": calendar_year,
                "baseline_rate": baseline_total,
                "scenario_rate": scenario_total,
            }
        )

    return pd.DataFrame(rows)


def simulate_survival(
    frame: pd.DataFrame,
    initial_age: int,
    scenario_map: dict[str, dict[str, float | int]] | None = None,
    start_year: int = START_YEAR,
    trend_table: TrendTable | None = None,
) -> SimulationResult:
    """Simulate cohort survival under piecewise-constant annual hazards.

    Args:
        frame: Normalized simulator mortality DataFrame from
            ``get_simulator_frame``. The function uses its age buckets, cause
            codes, and rates as the 2021 baseline mortality schedule.
        initial_age: Cohort starting age selected by the app's ``Initial age``
            slider. Survival is initialized to 1.0 at this attained age.
        scenario_map: Optional normalized intervention map returned by
            :func:`build_scenario_map`. When omitted or empty, every cause uses
            the baseline 2021 rate for the matching age bucket. When provided,
            a cause's rate switches to its scenario replacement rate once the
            projected calendar year reaches ``effective_year``.
        start_year: Calendar year assigned to ``initial_age``. The default is
            :data:`START_YEAR`, which matches the source year used for baseline
            rates in the simulator frame.
        trend_table: Optional significant-slope lookup from
            ``build_trend_model``. When present, cause rates without active
            interventions are projected as linear changes from the 2021
            baseline and floored at zero.

    Returns:
        A :class:`SimulationResult` containing the annual survival projection,
        plot-ready survival curve, terminal-tail hazard values, remaining life
        expectancy, total life expectancy age, and modeled death-age
        percentiles for the cohort.
    """
    scenario_map = scenario_map or {}
    rate_table, cause_names = _build_rate_table(frame)
    cause_codes = sorted(cause_names)

    last_change_year = max([start_year] + [int(config["effective_year"]) for config in scenario_map.values()])
    latest_intervention_age = initial_age + max(0, last_change_year - start_year)
    if trend_table is None:
        tail_start_age = max(95, latest_intervention_age)
    else:
        tail_start_age = max(130, latest_intervention_age)

    rows: list[dict[str, float | int]] = []
    survival = 1.0
    remaining_life_expectancy = 0.0

    for age in range(initial_age, tail_start_age):
        calendar_year = start_year + (age - initial_age)
        total_rate = sum(
            _resolve_rate(age, calendar_year, code, rate_table, scenario_map, trend_table)
            for code in cause_codes
        )
        hazard = total_rate / RATE_SCALE
        next_survival = survival * exp(-hazard)
        interval_expectation = survival if hazard == 0 else survival * (1 - exp(-hazard)) / hazard
        remaining_life_expectancy += interval_expectation

        rows.append(
            {
                "age_start": age,
                "age_end": age + 1,
                "calendar_year": calendar_year,
                "total_rate": total_rate,
                "hazard": hazard,
                "survival_start": survival,
                "survival_end": next_survival,
            }
        )
        survival = next_survival

    tail_calendar_year = start_year + (tail_start_age - initial_age)
    tail_total_rate = sum(
        _resolve_rate(tail_start_age, tail_calendar_year, code, rate_table, scenario_map, trend_table)
        for code in cause_codes
    )
    tail_hazard = tail_total_rate / RATE_SCALE

    if tail_hazard == 0:
        remaining_life_expectancy = inf if survival > 0 else remaining_life_expectancy
    else:
        remaining_life_expectancy += survival / tail_hazard

    annual_projection = pd.DataFrame(rows)
    percentiles = _compute_percentiles(annual_projection, tail_start_age, tail_hazard, survival)
    survival_curve = _build_survival_curve(annual_projection, tail_start_age, tail_hazard, survival, percentiles)

    return SimulationResult(
        annual_projection=annual_projection,
        survival_curve=survival_curve,
        tail_start_age=tail_start_age,
        tail_hazard=tail_hazard,
        survival_at_tail=survival,
        remaining_life_expectancy=remaining_life_expectancy,
        life_expectancy_age=initial_age + remaining_life_expectancy if remaining_life_expectancy != inf else inf,
        percentiles=percentiles,
    )


def _compute_percentiles(
    annual_projection: pd.DataFrame,
    tail_start_age: int,
    tail_hazard: float,
    survival_at_tail: float,
) -> dict[str, float | None]:
    """Return ages at which the cohort reaches each cumulative-death percentile.

    Args:
        annual_projection: DataFrame built inside :func:`simulate_survival`.
            Each row describes one explicit one-year interval before the
            terminal tail and must include ``age_start``, ``age_end``,
            ``hazard``, ``survival_start``, and ``survival_end``.
        tail_start_age: First age handled by the open-ended terminal tail. It
            is computed in :func:`simulate_survival` from the source data's
            ``95+`` bucket and any later intervention effective years.
        tail_hazard: Constant annual hazard used after ``tail_start_age``. It
            is computed in :func:`simulate_survival` by resolving all cause
            rates at the tail boundary and dividing their total by
            :data:`RATE_SCALE`.
        survival_at_tail: Survival probability at ``tail_start_age``. It is the
            final running survival value from :func:`simulate_survival` before
            the tail contribution is added.

    Returns:
        Mapping from percentile labels in :data:`PERCENTILES` to attained ages.
        Values are ``None`` when a target survival level is never crossed,
        which can only happen if the terminal tail hazard is zero.
    """
    percentiles: dict[str, float | None] = {}

    for label, death_fraction in PERCENTILES.items():
        # A death percentile is easier to find on the survival curve as its
        # complement. For p75 death, the target is 25% survival remaining.
        target_survival = 1.0 - death_fraction
        age_at_percentile: float | None = None

        for row in annual_projection.itertuples(index=False):
            if row.survival_start >= target_survival >= row.survival_end:
                # Within each modeled year the hazard is constant, so survival
                # follows S(t) = S0 * exp(-hazard * t). Solving for t gives
                # t = -log(target / S0) / hazard, measured from age_start.
                if row.hazard == 0:
                    # A zero-hazard interval is flat. This case can only match
                    # exactly, so returning the interval end avoids inventing a
                    # false within-year crossing time.
                    age_at_percentile = float(row.age_end)
                elif target_survival == row.survival_start:
                    age_at_percentile = float(row.age_start)
                else:
                    years_after_interval_start = (
                        -log(target_survival / row.survival_start) / row.hazard
                    )
                    age_at_percentile = float(row.age_start) + years_after_interval_start
                break

        if age_at_percentile is None:
            # After the explicit annual projection, the simulation assumes the
            # terminal age bucket continues forever at one constant tail hazard.
            if target_survival > survival_at_tail:
                # The curve is already below the target at the tail boundary.
                # Clamp to the boundary rather than extrapolating backward.
                age_at_percentile = float(tail_start_age)
            elif tail_hazard == 0:
                # With no tail hazard, survival never declines further, so any
                # unreached percentile is genuinely not reached by the model.
                age_at_percentile = None
            else:
                # Same inverse-survival calculation as above, but t is measured
                # from the start of the open-ended tail interval.
                years_after_tail_start = (
                    -log(target_survival / survival_at_tail) / tail_hazard
                )
                age_at_percentile = float(tail_start_age) + years_after_tail_start

        percentiles[label] = age_at_percentile

    return percentiles


def _build_survival_curve(
    annual_projection: pd.DataFrame,
    tail_start_age: int,
    tail_hazard: float,
    survival_at_tail: float,
    percentiles: dict[str, float | None],
) -> pd.DataFrame:
    """Build a chart-ready survival curve from annual rows and tail settings.

    Args:
        annual_projection: DataFrame built by :func:`simulate_survival`. It
            supplies the explicit survival points from ``initial_age`` through
            the last one-year interval before the terminal tail.
        tail_start_age: First age governed by the terminal tail. This comes
            from :func:`simulate_survival` and marks where the model stops
            creating explicit one-year rows.
        tail_hazard: Constant annual hazard after ``tail_start_age``. It comes
            from the total resolved mortality rate at the tail boundary divided
            by :data:`RATE_SCALE`.
        survival_at_tail: Survival probability at ``tail_start_age``. It is the
            value used as the starting point for tail extrapolation.
        percentiles: Percentile-age mapping returned by
            :func:`_compute_percentiles`. The farthest finite percentile helps
            determine how far the plotted curve should extend.

    Returns:
        DataFrame with ``age`` and ``survival`` columns. The chart layer passes
        this directly to Plotly when rendering baseline and scenario survival
        curves.
    """
    ages: list[float] = []
    survivals: list[float] = []

    if annual_projection.empty:
        initial_age = tail_start_age
    else:
        initial_age = int(annual_projection.iloc[0]["age_start"])

    ages.append(float(initial_age))
    survivals.append(1.0)

    for row in annual_projection.itertuples(index=False):
        ages.append(float(row.age_end))
        survivals.append(float(row.survival_end))

    finite_percentiles = [age for age in percentiles.values() if age is not None]
    curve_max_age = int(max([tail_start_age + 12, *(finite_percentiles or [tail_start_age + 12])]))
    curve_max_age = min(curve_max_age + 2, max(130, tail_start_age + 20))

    for age in range(max(tail_start_age + 1, int(ages[-1]) + 1), curve_max_age + 1):
        years_in_tail = age - tail_start_age
        tail_survival = survival_at_tail if tail_hazard == 0 else survival_at_tail * exp(-tail_hazard * years_in_tail)
        ages.append(float(age))
        survivals.append(float(tail_survival))

    return pd.DataFrame({"age": ages, "survival": survivals})
