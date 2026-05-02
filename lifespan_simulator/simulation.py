"""Piecewise-constant mortality simulation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, inf, log
from typing import Any

import pandas as pd

from .data import get_age_bucket_start

START_YEAR = 2021
RATE_SCALE = 100_000.0
PERCENTILES = {
    "p50": 0.50,
    "p75": 0.75,
    "p90": 0.90,
    "p95": 0.95,
    "p99": 0.99,
}


@dataclass(frozen=True)
class SimulationResult:
    """Full result for a single cohort simulation."""

    annual_projection: pd.DataFrame
    survival_curve: pd.DataFrame
    tail_start_age: int
    tail_hazard: float
    survival_at_tail: float
    remaining_life_expectancy: float
    life_expectancy_age: float
    percentiles: dict[str, float | None]


def _build_rate_table(frame: pd.DataFrame) -> tuple[dict[int, dict[str, float]], dict[str, str]]:
    rate_table = (
        frame.pivot_table(index="age_start", columns="cause_code", values="rate", aggfunc="first")
        .fillna(0.0)
        .to_dict(orient="index")
    )
    cause_names = frame.drop_duplicates("cause_code").set_index("cause_code")["cause_name"].to_dict()
    return rate_table, cause_names


def build_scenario_map(raw_scenarios: dict[str, dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    """Normalize UI scenario inputs."""
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
) -> float:
    bucket_start = get_age_bucket_start(age)
    baseline_rate = float(rate_table[bucket_start].get(cause_code, 0.0))
    scenario = scenario_map.get(cause_code)
    if scenario and calendar_year >= int(scenario["effective_year"]):
        return float(scenario["rate"])
    return baseline_rate


def project_cause_rates(
    frame: pd.DataFrame,
    initial_age: int,
    selected_causes: list[str],
    scenario_map: dict[str, dict[str, float | int]],
    max_age: int,
    start_year: int = START_YEAR,
) -> pd.DataFrame:
    """Project baseline and scenario rates by attained age for selected causes."""
    rate_table, cause_names = _build_rate_table(frame)
    rows: list[dict[str, Any]] = []

    for age in range(initial_age, max_age + 1):
        calendar_year = start_year + (age - initial_age)
        for cause_code in selected_causes:
            bucket_start = get_age_bucket_start(age)
            rows.append(
                {
                    "age": age,
                    "calendar_year": calendar_year,
                    "cause_code": cause_code,
                    "cause_name": cause_names[cause_code],
                    "baseline_rate": float(rate_table[bucket_start].get(cause_code, 0.0)),
                    "scenario_rate": _resolve_rate(age, calendar_year, cause_code, rate_table, scenario_map),
                }
            )

    return pd.DataFrame(rows)


def project_total_rates(
    frame: pd.DataFrame,
    initial_age: int,
    scenario_map: dict[str, dict[str, float | int]],
    max_age: int,
    start_year: int = START_YEAR,
) -> pd.DataFrame:
    """Project total mortality rates by attained age."""
    rate_table, cause_names = _build_rate_table(frame)
    cause_codes = sorted(cause_names)
    rows: list[dict[str, Any]] = []

    for age in range(initial_age, max_age + 1):
        calendar_year = start_year + (age - initial_age)
        bucket_start = get_age_bucket_start(age)
        baseline_total = sum(float(rate_table[bucket_start].get(code, 0.0)) for code in cause_codes)
        scenario_total = sum(
            _resolve_rate(age, calendar_year, code, rate_table, scenario_map) for code in cause_codes
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
) -> SimulationResult:
    """Simulate a synthetic cohort under piecewise-constant annual hazards."""
    scenario_map = scenario_map or {}
    rate_table, cause_names = _build_rate_table(frame)
    cause_codes = sorted(cause_names)

    last_change_year = max([start_year] + [int(config["effective_year"]) for config in scenario_map.values()])
    tail_start_age = max(95, initial_age + max(0, last_change_year - start_year))

    rows: list[dict[str, float | int]] = []
    survival = 1.0
    remaining_life_expectancy = 0.0

    for age in range(initial_age, tail_start_age):
        calendar_year = start_year + (age - initial_age)
        total_rate = sum(_resolve_rate(age, calendar_year, code, rate_table, scenario_map) for code in cause_codes)
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
        _resolve_rate(tail_start_age, tail_calendar_year, code, rate_table, scenario_map) for code in cause_codes
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
    """Return ages at which the cohort reaches each cumulative-death percentile."""
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
    curve_max_age = min(curve_max_age + 2, 130)

    for age in range(max(tail_start_age + 1, int(ages[-1]) + 1), curve_max_age + 1):
        years_in_tail = age - tail_start_age
        tail_survival = survival_at_tail if tail_hazard == 0 else survival_at_tail * exp(-tail_hazard * years_in_tail)
        ages.append(float(age))
        survivals.append(float(tail_survival))

    return pd.DataFrame({"age": ages, "survival": survivals})
