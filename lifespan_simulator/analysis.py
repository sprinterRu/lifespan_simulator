"""Reference checks and deterministic sensitivity experiments."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .config import PandemicConfig, ProjectionConfig
from .interventions import ScenarioMap, build_scenario_map
from .mortality import MortalityModel
from .simulation import simulate_survival


def compare_cause_totals(frame: pd.DataFrame, official_totals: pd.DataFrame) -> pd.DataFrame:
    """Audit the unmodified CSV before any life-table calibration."""
    totals = frame.groupby(["age_code", "age_start", "age_display"], as_index=False).rate.sum()
    audit = totals.merge(
        official_totals[["age_code", "rate"]],
        on="age_code",
        how="outer",
        suffixes=("_csv", "_official"),
        validate="one_to_one",
    )
    if audit[["rate_csv", "rate_official"]].isna().any().any():
        raise ValueError("Official and source cause totals have different age coverage")
    audit["difference"] = audit.rate_csv - audit.rate_official
    audit["difference_pct"] = 100 * audit.difference / audit.rate_official
    return audit.sort_values("age_start").reset_index(drop=True)


def compare_life_expectancy(frame: pd.DataFrame, official: pd.DataFrame) -> pd.DataFrame:
    """Compare period e(x) with no secular trend and the observed pandemic level.

    These checks validate reconstruction/calibration, not future predictive skill.
    Open-age LIFEXP is remaining life from 95; open-age Mx is not M(95).
    """
    model = MortalityModel(frame, official_life_table=official)
    fixed = ProjectionConfig(pandemic=PandemicConfig(mode="hold_2021"), trend_horizon_years=0)
    reference = official[official.indicator == "LIFEXP"].set_index("age").value
    rows = []
    for age in (35, 65, 85, 95):
        row = {
            "age": age,
            "official_remaining_years": float(reference.loc[str(age) if age < 95 else "95+"]),
        }
        for label, config in [
            ("uncalibrated_flat", replace(fixed, calibrate=False, old_age_model="flat")),
            ("uncalibrated_kannisto", replace(fixed, calibrate=False)),
            ("calibrated_kannisto", fixed),
        ]:
            row[label] = simulate_survival(model, age, config=config).remaining_life_expectancy
        row["calibrated_difference"] = row["calibrated_kannisto"] - row["official_remaining_years"]
        rows.append(row)
    return pd.DataFrame(rows)


def sensitivity_variants(
    config: ProjectionConfig, scenarios: ScenarioMap
) -> list[tuple[str, ProjectionConfig, ScenarioMap]]:
    """One assumption at a time, not a probability distribution or confidence band."""
    variants = [("Current assumptions", config, scenarios)]
    for label, pandemic in [
        ("COVID: hold 2021 level", PandemicConfig(mode="hold_2021")),
        ("COVID: fades to zero, half-life 3y", PandemicConfig(residual_fraction=0)),
        ("COVID: 30% residual, half-life 3y", PandemicConfig(residual_fraction=0.3)),
        ("COVID: 10% residual, half-life 1y", PandemicConfig(half_life_years=1)),
        ("COVID: 10% residual, half-life 10y", PandemicConfig(half_life_years=10)),
    ]:
        variants.append((label, replace(config, pandemic=pandemic), scenarios))
    for mode in ("flat", "gompertz", "kannisto"):
        if mode != config.old_age_model:
            variants.append((f"Old age: {mode}", replace(config, old_age_model=mode), scenarios))
    for slope in (0.75, 1.25):
        variants.append(
            (
                f"Old age: slope ×{slope:g}",
                replace(config, old_age_slope_multiplier=slope),
                scenarios,
            )
        )
    for floor in (0.05, 0.30):
        variants.append(
            (f"Trend floor: {floor:.0%}", replace(config, trend_floor=floor), scenarios)
        )
    for horizon in (20, 80):
        variants.append(
            (
                f"Trend horizon: {horizon}y after 2021",
                replace(config, trend_horizon_years=horizon),
                scenarios,
            )
        )
    variants.append(("No all-cause calibration", replace(config, calibrate=False), scenarios))
    if scenarios:
        for shift in (-10, 10):
            shifted = {
                code: value._replace(
                    slow_start_year=max(2021, min(2120, value.slow_start_year + shift)),
                    fast_start_year=max(2021, min(2120, value.fast_start_year + shift)),
                )
                for code, value in scenarios.items()
            }
            variants.append(
                (f"Interventions: dates {shift:+d}y", config, build_scenario_map(shifted))
            )
        for factor in (0.5, 1.5):
            changed = {
                code: value._replace(
                    slow_rate=min(1, value.slow_rate * factor),
                    fast_rate=min(1, value.fast_rate * factor),
                )
                for code, value in scenarios.items()
            }
            variants.append(
                (f"Interventions: reductions ×{factor:g}", config, build_scenario_map(changed))
            )
        for floor in (0.001, 0.05):
            changed = {code: value._replace(floor=floor) for code, value in scenarios.items()}
            variants.append(
                (f"Interventions: residual {floor:.1%}", config, build_scenario_map(changed))
            )
    return variants


def run_sensitivity(
    model: MortalityModel, initial_age: int, scenarios: ScenarioMap, config: ProjectionConfig
) -> pd.DataFrame:
    rows = []
    for label, assumptions, interventions in sensitivity_variants(config, scenarios):
        baseline = simulate_survival(model, initial_age, config=assumptions)
        result = (
            simulate_survival(model, initial_age, interventions, assumptions)
            if interventions
            else baseline
        )
        rows.append(
            {
                "Assumption": label,
                "Baseline mean age": baseline.life_expectancy_age,
                "Scenario mean age": result.life_expectancy_age,
                "Gain (years)": result.life_expectancy_age - baseline.life_expectancy_age
                if result.converged and baseline.converged
                else np.nan,
                "Median age": result.percentiles["p50"],
                "p99 age": result.percentiles["p99"],
                "Expected years before horizon": result.restricted_remaining_life_expectancy,
                "Alive at horizon (%)": 100 * result.survival_at_end,
                "Unintegrated years upper bound": result.remaining_expectation_bound,
                "Mean resolved": result.converged,
            }
        )
    return pd.DataFrame(rows)
