"""Validate intervention settings and evaluate their two-stage reduction paths."""

from __future__ import annotations

from math import isfinite
from typing import Any, NamedTuple

import numpy as np


class InterventionConfig(NamedTuple):
    slow_start_year: int
    slow_rate: float
    fast_start_year: int
    fast_rate: float
    floor: float


ScenarioMap = dict[str, InterventionConfig]


def build_scenario_map(raw_scenarios: dict[str, dict[str, Any] | tuple]) -> ScenarioMap:
    """Reject invalid inputs rather than silently changing the requested scenario."""
    scenarios = {}
    for code, values in raw_scenarios.items():
        config = (
            InterventionConfig(*values)
            if isinstance(values, tuple)
            else InterventionConfig(**values)
        )
        if any(not isfinite(float(value)) for value in config):
            raise ValueError(f"Intervention values must be finite: {code}")
        if any(
            int(year) != year or not 2021 <= year <= 2120
            for year in (config.slow_start_year, config.fast_start_year)
        ):
            raise ValueError("Intervention years must be integers between 2021 and 2120")
        if config.fast_start_year < config.slow_start_year:
            raise ValueError("Fast stage cannot precede slow stage")
        if not (
            0 <= config.slow_rate <= 1 and 0 <= config.fast_rate <= 1 and 1e-12 <= config.floor <= 1
        ):
            raise ValueError("Reduction rates must be in [0, 1] and floor in [1e-12, 1]")
        scenarios[str(code)] = InterventionConfig(
            int(config.slow_start_year),
            float(config.slow_rate),
            int(config.fast_start_year),
            float(config.fast_rate),
            float(config.floor),
        )
    return scenarios


def intervention_multiplier(years: np.ndarray, config: InterventionConfig) -> np.ndarray:
    """Include the first annual reduction in the specified start year."""
    slow_years = np.clip(
        years - config.slow_start_year + 1, 0, config.fast_start_year - config.slow_start_year
    )
    fast_years = np.maximum(0, years - config.fast_start_year + 1)

    def stage(rate: float, duration: np.ndarray) -> np.ndarray:
        if rate == 1:
            return np.where(duration > 0, 0.0, 1.0)
        return np.exp(np.log1p(-rate) * duration)

    return np.maximum(
        config.floor, stage(config.slow_rate, slow_years) * stage(config.fast_rate, fast_years)
    )


def limiting_intervention_multiplier(config: InterventionConfig) -> float:
    """A zero fast rate preserves the completed slow stage instead of reaching the floor."""
    if config.fast_rate > 0:
        return config.floor
    return float(intervention_multiplier(np.array([config.fast_start_year]), config)[0])
