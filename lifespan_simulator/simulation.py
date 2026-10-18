"""Integrate a cohort's survival under piecewise-constant annual hazards.

Projection is bounded in work, not by an assumed maximum human lifespan. If
survivors beyond the numerical horizon can materially change the mean, report
only the restricted mean and mark full life expectancy as unresolved.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, expm1, log

import numpy as np
import pandas as pd

from .config import BASE_YEAR, RATE_SCALE, ProjectionConfig
from .interventions import ScenarioMap, build_scenario_map
from .mortality import MortalityModel

START_YEAR = BASE_YEAR
PERCENTILES = {"p50": 0.50, "p75": 0.75, "p90": 0.90, "p95": 0.95, "p99": 0.99}


@dataclass(frozen=True)
class SimulationResult:
    annual_projection: pd.DataFrame
    survival_curve: pd.DataFrame
    integration_end_age: int
    survival_at_end: float
    restricted_remaining_life_expectancy: float
    remaining_life_expectancy: float | None
    life_expectancy_age: float | None
    remaining_expectation_bound: float
    percentiles: dict[str, float | None]

    @property
    def converged(self) -> bool:
        return self.remaining_life_expectancy is not None


def project_cause_rates(
    model: MortalityModel,
    initial_age: int,
    selected_causes: list[str],
    scenario_map: ScenarioMap,
    max_age: int,
    config: ProjectionConfig | None = None,
) -> pd.DataFrame:
    config = config or ProjectionConfig()
    ages, years = _age_year_grid(model, initial_age, max_age, config)
    baseline = model.rates(ages, years, config)
    scenario = model.rates(ages, years, config, scenario_map) if scenario_map else baseline
    indices = [model.codes.index(code) for code in selected_causes]
    return pd.DataFrame(
        {
            "age": np.repeat(ages, len(indices)),
            "calendar_year": np.repeat(years, len(indices)),
            "cause_code": np.tile(selected_causes, len(ages)),
            "cause_name": np.tile([model.names[code] for code in selected_causes], len(ages)),
            "baseline_rate": baseline[:, indices].ravel(),
            "scenario_rate": scenario[:, indices].ravel(),
        }
    )


def project_total_rates(
    model: MortalityModel,
    initial_age: int,
    scenario_map: ScenarioMap,
    max_age: int,
    config: ProjectionConfig | None = None,
) -> pd.DataFrame:
    config = config or ProjectionConfig()
    ages, years = _age_year_grid(model, initial_age, max_age, config)
    baseline = model.rates(ages, years, config).sum(axis=1)
    scenario = (
        model.rates(ages, years, config, scenario_map).sum(axis=1) if scenario_map else baseline
    )
    return pd.DataFrame(
        {"age": ages, "calendar_year": years, "baseline_rate": baseline, "scenario_rate": scenario}
    )


def _age_year_grid(
    model: MortalityModel, initial_age: int, end_age: int, config: ProjectionConfig
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(initial_age, int) or not model.minimum_age <= initial_age <= 95:
        raise ValueError(f"Initial age must be an integer between {model.minimum_age} and 95")
    if not initial_age <= end_age <= 500:
        raise ValueError("Projection end age must lie between the initial age and 500")
    ages = np.arange(initial_age, end_age + 1)
    return ages, config.start_year + ages - initial_age


def simulate_survival(
    model: MortalityModel,
    initial_age: int,
    scenario_map: ScenarioMap | None = None,
    config: ProjectionConfig | None = None,
) -> SimulationResult:
    config = config or ProjectionConfig()
    scenarios = build_scenario_map(scenario_map or {})
    ages, years = _age_year_grid(model, initial_age, config.max_age, config)
    hazards = model.rates(ages, years, config, scenarios).sum(axis=1) / RATE_SCALE
    survival = 1.0
    expectation = 0.0
    rows = []
    percentiles: dict[str, float | None] = dict.fromkeys(PERCENTILES)

    # Integrate through max_age, excluding the following [max_age, max_age+1) year.
    # expm1 avoids cancellation when interventions make the hazard very small.
    for age, year, hazard in zip(ages[:-1], years[:-1], hazards[:-1], strict=True):
        next_survival = survival * exp(-hazard)
        expectation += survival if hazard == 0 else survival * -expm1(-hazard) / hazard
        for label, fraction in PERCENTILES.items():
            target = 1 - fraction
            if percentiles[label] is None and survival >= target >= next_survival and hazard > 0:
                percentiles[label] = float(age - log(target / survival) / hazard)
        rows.append(
            {
                "age_start": int(age),
                "age_end": int(age + 1),
                "calendar_year": int(year),
                "total_rate": hazard * RATE_SCALE,
                "hazard": hazard,
                "survival_start": survival,
                "survival_end": next_survival,
            }
        )
        survival = next_survival

    lower_hazard = model.future_hazard_lower_bound(
        config.max_age, int(years[-1]), config, scenarios
    )
    remainder_bound = survival / lower_hazard if lower_hazard > 0 else float("inf")
    converged = remainder_bound <= config.expectation_tolerance
    annual = pd.DataFrame(rows)
    curve = _survival_curve(annual)
    return SimulationResult(
        annual_projection=annual,
        survival_curve=curve,
        integration_end_age=config.max_age,
        survival_at_end=survival,
        restricted_remaining_life_expectancy=expectation,
        remaining_life_expectancy=expectation if converged else None,
        life_expectancy_age=initial_age + expectation if converged else None,
        remaining_expectation_bound=remainder_bound,
        percentiles=percentiles,
    )


def _survival_curve(annual: pd.DataFrame) -> pd.DataFrame:
    """Sample the within-year exponential, keeping the entire numerical horizon."""
    fractions = np.array([0.0, 0.25, 0.50, 0.75])
    ages = annual.age_start.to_numpy()[:, None] + fractions
    survival = annual.survival_start.to_numpy()[:, None] * np.exp(
        -annual.hazard.to_numpy()[:, None] * fractions
    )
    return pd.DataFrame(
        {
            "age": np.append(ages.ravel(), annual.age_end.iloc[-1]),
            "survival": np.append(survival.ravel(), annual.survival_end.iloc[-1]),
        }
    )
