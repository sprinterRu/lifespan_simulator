"""Piecewise-constant mortality simulation utilities.

The functions in this module receive the normalized 2021 simulator dataset
prepared in :mod:`lifespan_simulator.data` and optional intervention settings
collected by the Streamlit UI. They convert cause-specific mortality rates into
annual hazards by interpolating source age-bucket rates to attained ages, apply
any selected two-stage exponential interventions, and return the tables needed
by the app's charts and metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, exp, inf, log
from typing import Any, NamedTuple

import pandas as pd

START_YEAR = 2021
# Mortality rates in the source data are expressed per 100,000 population.
# Dividing by this value converts a rate into the annual hazard used by the
# exponential survival model.
RATE_SCALE = 100_000.0
# With trend dampening enabled, fitted annual slopes are applied fully for the
# first projection window, linearly tapered to zero during the second window,
# and then held fixed.
TREND_FULL_YEARS = 15
TREND_TAPER_YEARS = 15
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


class InterventionConfig(NamedTuple):
    """Five-number intervention tuple used by scenario projections."""

    slow_start_year: int
    slow_rate: float
    fast_start_year: int
    fast_rate: float
    floor: float


ScenarioMap = dict[str, InterventionConfig]


class AgeRatePoint(NamedTuple):
    """One source age-bucket point used for annual age interpolation."""

    age_mid: float
    age_start: int
    age_end: int | None
    rate: float


RateTable = dict[str, tuple[AgeRatePoint, ...]]


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
            computed by :func:`simulate_survival` from age 95, any explicit
            trend horizon, and any later year when active interventions reach
            their mortality floor.
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


def _build_rate_table(frame: pd.DataFrame) -> tuple[RateTable, dict[str, str]]:
    """Convert the simulator mortality frame into fast lookup dictionaries.

    Args:
        frame: Normalized simulator mortality DataFrame. In the app this is
            ``simulator_frame`` returned by
            ``get_simulator_frame(load_mortality_data(DATA_PATH))`` in
            ``app.load_frames``. It must contain one 2021 row per age bucket
            and cause, with at least ``age_start``, ``cause_code``,
            ``cause_name``, and ``rate`` columns.

    Returns:
        A pair of dictionaries. The first maps each cause code to sorted
        age-midpoint rate observations used for annual interpolation. The
        second maps each cause code to the human-readable cause name used in
        tables and charts.
    """
    rate_table = {
        str(cause_code): tuple(
            AgeRatePoint(
                age_mid=float(row.age_mid),
                age_start=int(row.age_start),
                age_end=None if pd.isna(row.age_end) else int(row.age_end),
                rate=float(row.rate),
            )
            for row in group.sort_values("age_mid").itertuples(index=False)
        )
        for cause_code, group in frame.groupby("cause_code", sort=False)
    }
    cause_names = frame.drop_duplicates("cause_code").set_index("cause_code")["cause_name"].to_dict()
    return rate_table, cause_names


def build_scenario_map(raw_scenarios: dict[str, dict[str, Any] | tuple[Any, ...]]) -> ScenarioMap:
    """Normalize raw intervention inputs into the simulation scenario format.

    Args:
        raw_scenarios: Mapping keyed by cause code. In the app it is assembled
            in ``render_simulator_tab`` from the selected intervention causes
            and the Streamlit number inputs for the five intervention values:
            slow-stage start year, slow annual fractional reduction, fast-stage
            start year, fast annual fractional reduction, and irreducible floor
            multiplier. Each value may be a mapping with named fields or a
            five-item tuple in that order.

    Returns:
        A clean scenario map keyed by cause code. Values are
        :class:`InterventionConfig` tuples. Rates are clamped to the
        ``[0, 1]`` interval, the floor is clamped to ``(0, 1]``, and the fast
        start year is not allowed to precede the slow start year.
    """
    scenario_map: ScenarioMap = {}
    for cause_code, config in raw_scenarios.items():
        if isinstance(config, tuple):
            slow_start_year, slow_rate, fast_start_year, fast_rate, floor = config
        else:
            slow_start_year = config["slow_start_year"]
            slow_rate = config["slow_rate"]
            fast_start_year = config["fast_start_year"]
            fast_rate = config["fast_rate"]
            floor = config["floor"]

        normalized_slow_start = int(slow_start_year)
        normalized_fast_start = max(normalized_slow_start, int(fast_start_year))
        scenario_map[cause_code] = InterventionConfig(
            slow_start_year=normalized_slow_start,
            slow_rate=_clamp_unit_interval(float(slow_rate)),
            fast_start_year=normalized_fast_start,
            fast_rate=_clamp_unit_interval(float(fast_rate)),
            floor=min(1.0, max(1e-12, float(floor))),
        )
    return scenario_map


def _clamp_unit_interval(value: float) -> float:
    """Clamp a fractional reduction rate to the closed unit interval."""
    return min(1.0, max(0.0, value))


def _counterfactual_rate(
    age: int,
    calendar_year: int,
    cause_code: str,
    rate_table: RateTable,
    trend_table: TrendTable | None = None,
    trend_dampening: bool = False,
) -> float:
    """Resolve the no-intervention rate before applying a scenario multiplier."""
    baseline_rate = _interpolate_rate(age, rate_table.get(cause_code, ()))
    if trend_table is not None:
        slope = _interpolate_trend_slope(age, cause_code, rate_table.get(cause_code, ()), trend_table)
        years_since_start = calendar_year - START_YEAR
        trend_years = (
            dampened_trend_years(years_since_start)
            if trend_dampening
            else float(years_since_start)
        )
        return max(0.0, baseline_rate + slope * trend_years)
    return baseline_rate


def _interpolate_rate(age: int | float, points: tuple[AgeRatePoint, ...]) -> float:
    """Linearly interpolate a source age-bucket rate to an annual age."""
    return _interpolate_age_value(
        age,
        [(point.age_mid, point.rate) for point in points],
        terminal_start_age=_terminal_bucket_start(points),
    )


def _interpolate_trend_slope(
    age: int | float,
    cause_code: str,
    points: tuple[AgeRatePoint, ...],
    trend_table: TrendTable,
) -> float:
    """Linearly interpolate age-bucket trend slopes to an annual age."""
    return _interpolate_age_value(
        age,
        [
            (point.age_mid, float(trend_table.get(point.age_start, {}).get(cause_code, 0.0)))
            for point in points
        ],
        terminal_start_age=_terminal_bucket_start(points),
    )


def _terminal_bucket_start(points: tuple[AgeRatePoint, ...]) -> int | None:
    """Return the source start age for an open-ended terminal bucket."""
    if points and points[-1].age_end is None:
        return points[-1].age_start
    return None


def _interpolate_age_value(
    age: int | float,
    points: list[tuple[float, float]],
    terminal_start_age: int | None = None,
) -> float:
    """Interpolate over age, while holding an open-ended terminal bucket flat."""
    if not points:
        return 0.0

    numeric_age = float(age)
    if terminal_start_age is not None and numeric_age >= terminal_start_age:
        return points[-1][1]
    if numeric_age <= points[0][0]:
        return points[0][1]
    if numeric_age >= points[-1][0]:
        return points[-1][1]

    for (lower_age, lower_value), (upper_age, upper_value) in zip(points, points[1:]):
        if lower_age <= numeric_age <= upper_age:
            position = (numeric_age - lower_age) / (upper_age - lower_age)
            return lower_value + (upper_value - lower_value) * position

    return points[-1][1]


def _intervention_multiplier(calendar_year: int, scenario: InterventionConfig) -> float:
    """Return the two-stage exponential multiplier for one calendar year."""
    if calendar_year < scenario.slow_start_year:
        return 1.0

    if calendar_year < scenario.fast_start_year:
        slow_years = calendar_year - scenario.slow_start_year + 1
        multiplier = (1.0 - scenario.slow_rate) ** slow_years
    else:
        slow_years = max(0, scenario.fast_start_year - scenario.slow_start_year)
        fast_years = calendar_year - scenario.fast_start_year + 1
        multiplier = ((1.0 - scenario.slow_rate) ** slow_years) * (
            (1.0 - scenario.fast_rate) ** fast_years
        )

    return max(scenario.floor, multiplier)


def _intervention_stabilization_year(scenario: InterventionConfig) -> int:
    """Find the first year when an intervention multiplier stops changing."""
    if scenario.floor >= 1.0 or (scenario.slow_rate <= 0.0 and scenario.fast_rate <= 0.0):
        return START_YEAR

    slow_stage_years = max(0, scenario.fast_start_year - scenario.slow_start_year)
    if slow_stage_years > 0:
        years_to_floor = _years_to_reach_floor(1.0, scenario.slow_rate, scenario.floor)
        if years_to_floor is not None and years_to_floor <= slow_stage_years:
            return scenario.slow_start_year + years_to_floor - 1
        multiplier_at_fast_start = (1.0 - scenario.slow_rate) ** slow_stage_years
    else:
        multiplier_at_fast_start = 1.0

    years_to_floor = _years_to_reach_floor(
        multiplier_at_fast_start,
        scenario.fast_rate,
        scenario.floor,
    )
    if years_to_floor is None:
        return scenario.fast_start_year
    return scenario.fast_start_year + years_to_floor - 1


def _years_to_reach_floor(
    current_multiplier: float,
    annual_reduction: float,
    floor: float,
) -> int | None:
    """Return the number of annual reductions needed to reach ``floor``."""
    if current_multiplier <= floor:
        return 0
    if annual_reduction <= 0.0:
        return None
    if annual_reduction >= 1.0:
        return 1
    return max(1, int(ceil(log(floor / current_multiplier) / log(1.0 - annual_reduction))))


def _resolve_rate(
    age: int,
    calendar_year: int,
    cause_code: str,
    rate_table: RateTable,
    scenario_map: ScenarioMap,
    trend_table: TrendTable | None = None,
    trend_dampening: bool = False,
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
            It provides source 2021 mortality rates by cause code, with
            age-bucket midpoint observations used for annual interpolation.
        scenario_map: Normalized intervention map returned by
            :func:`build_scenario_map`. If it contains ``cause_code``, the
            cause's no-intervention rate is multiplied by the selected
            two-stage exponential intervention path.
        trend_table: Optional nested lookup returned by
            ``lifespan_simulator.trends.build_trend_model``. When present, it
            supplies significant linear annual slopes by age bucket and cause
            code. Non-significant cause-age trends are represented as ``0.0``.
        trend_dampening: When true, the fitted slope is applied fully for
            :data:`TREND_FULL_YEARS`, tapered to zero over
            :data:`TREND_TAPER_YEARS`, and then held fixed.

    Returns:
        The applicable mortality rate per 100,000 population. Historical trend
        adjustment is resolved first; active interventions then reduce that
        counterfactual rate by a floor-limited exponential multiplier.
    """
    counterfactual_rate = _counterfactual_rate(
        age,
        calendar_year,
        cause_code,
        rate_table,
        trend_table,
        trend_dampening,
    )
    scenario = scenario_map.get(cause_code)
    if scenario is None:
        return counterfactual_rate
    return counterfactual_rate * _intervention_multiplier(calendar_year, scenario)


def dampened_trend_years(
    years_since_start: int | float,
    full_years: int = TREND_FULL_YEARS,
    taper_years: int = TREND_TAPER_YEARS,
) -> float:
    """Return the cumulative trend exposure after linear slope tapering.

    A fitted annual slope is applied at full strength through ``full_years``.
    During the following ``taper_years``, the marginal slope contribution fades
    linearly to zero. After that, the cumulative trend contribution is frozen.
    """
    years = float(years_since_start)
    if years <= full_years:
        return years
    if taper_years <= 0:
        return float(full_years)

    taper_elapsed = min(years - full_years, taper_years)
    dampened_years = full_years + taper_elapsed - (taper_elapsed**2 / (2 * taper_years))

    if years > full_years + taper_years:
        return full_years + (taper_years / 2)
    return dampened_years


def project_cause_rates(
    frame: pd.DataFrame,
    initial_age: int,
    selected_causes: list[str],
    scenario_map: ScenarioMap,
    max_age: int,
    start_year: int = START_YEAR,
    trend_table: TrendTable | None = None,
    trend_dampening: bool = False,
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
        trend_dampening: When true, trend adjustments are tapered and then
            frozen instead of extrapolating linearly forever.

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
                        trend_dampening,
                    ),
                    "scenario_rate": _resolve_rate(
                        age,
                        calendar_year,
                        cause_code,
                        rate_table,
                        scenario_map,
                        trend_table,
                        trend_dampening,
                    ),
                }
            )

    return pd.DataFrame(rows)


def project_total_rates(
    frame: pd.DataFrame,
    initial_age: int,
    scenario_map: ScenarioMap,
    max_age: int,
    start_year: int = START_YEAR,
    trend_table: TrendTable | None = None,
    trend_dampening: bool = False,
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
            :func:`build_scenario_map`. Each matching cause applies a
            floor-limited two-stage exponential multiplier to the otherwise
            projected cause-specific rate.
        max_age: Last attained age to project. The app typically passes the
            display horizon returned by ``get_display_horizon(initial_age)``.
        start_year: Calendar year assigned to ``initial_age``. It defaults to
            :data:`START_YEAR`, the same year used to filter the simulator
            mortality frame.
        trend_table: Optional significant-slope lookup from
            ``build_trend_model``. When present, both total series include
            historical trend adjustments, and the scenario series additionally
            applies active interventions.
        trend_dampening: When true, trend adjustments are tapered and then
            frozen instead of extrapolating linearly forever.

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
            _resolve_rate(age, calendar_year, code, rate_table, {}, trend_table, trend_dampening)
            for code in cause_codes
        )
        scenario_total = sum(
            _resolve_rate(age, calendar_year, code, rate_table, scenario_map, trend_table, trend_dampening)
            for code in cause_codes
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
    scenario_map: ScenarioMap | None = None,
    start_year: int = START_YEAR,
    trend_table: TrendTable | None = None,
    trend_dampening: bool = False,
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
            the no-intervention rate for the matching age bucket. When
            provided, selected causes follow the scenario's slow-stage and
            fast-stage exponential mortality reductions down to their floor.
        start_year: Calendar year assigned to ``initial_age``. The default is
            :data:`START_YEAR`, which matches the source year used for baseline
            rates in the simulator frame.
        trend_table: Optional significant-slope lookup from
            ``build_trend_model``. When present, cause rates without active
            interventions are projected as linear changes from the 2021
            baseline and floored at zero.
        trend_dampening: When true, trend adjustments are tapered and then
            frozen instead of extrapolating linearly forever.

    Returns:
        A :class:`SimulationResult` containing the annual survival projection,
        plot-ready survival curve, terminal-tail hazard values, remaining life
        expectancy, total life expectancy age, and modeled death-age
        percentiles for the cohort.
    """
    scenario_map = scenario_map or {}
    rate_table, cause_names = _build_rate_table(frame)
    cause_codes = sorted(cause_names)

    last_stable_year = max(
        [start_year] + [_intervention_stabilization_year(config) for config in scenario_map.values()]
    )
    latest_intervention_age = initial_age + max(0, last_stable_year - start_year)
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
            _resolve_rate(
                age,
                calendar_year,
                code,
                rate_table,
                scenario_map,
                trend_table,
                trend_dampening,
            )
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
        _resolve_rate(
            tail_start_age,
            tail_calendar_year,
            code,
            rate_table,
            scenario_map,
            trend_table,
            trend_dampening,
        )
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
            ``95+`` bucket, any explicit trend horizon, and any later year
            when active intervention multipliers stop changing.
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
