import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from lifespan_simulator.analysis import (
    compare_cause_totals,
    compare_life_expectancy,
    run_sensitivity,
)
from lifespan_simulator.config import PandemicConfig, ProjectionConfig
from lifespan_simulator.data import REFERENCE_DIR, load_official_cause_totals
from lifespan_simulator.mortality import MortalityModel
from lifespan_simulator.simulation import simulate_survival
from lifespan_simulator.trends import build_trend_model


def test_reference_snapshot_checksums():
    manifest = json.loads((REFERENCE_DIR / "reference_metadata.json").read_text())
    for file in manifest["files"]:
        assert (
            hashlib.sha256((REFERENCE_DIR / file["file"]).read_bytes()).hexdigest()
            == file["sha256"]
        )


def test_cause_totals_remain_auditable(inputs):
    audit = compare_cause_totals(inputs[1], load_official_cause_totals())
    assert len(audit) == 13
    assert audit.difference.abs().max() < 4
    assert 0.5 < audit.difference_pct.abs().max() < 0.6


def test_calibrated_period_life_expectancy_matches_reference(inputs):
    comparison = compare_life_expectancy(inputs[1], inputs[2])
    assert (comparison.loc[comparison.age < 95, "calibrated_difference"].abs() < 0.1).all()
    # 95+ is extrapolated, and the discrepancy must stay visible rather than
    # being fitted away using the same validation statistic.
    assert comparison.loc[comparison.age == 95, "calibrated_difference"].abs().iloc[0] < 0.3


def test_calibration_preserves_cause_shares_and_matches_mx(inputs):
    _, frame, official = inputs
    model = MortalityModel(frame, official_life_table=official)
    ages = np.arange(35, 95)
    config = ProjectionConfig(pandemic=PandemicConfig(mode="hold_2021"))
    rates = model.rates(ages, np.full(len(ages), 2021), config)
    raw = model.rates(ages, np.full(len(ages), 2021), replace(config, calibrate=False))
    np.testing.assert_allclose(rates / rates.sum(axis=1)[:, None], raw / raw.sum(axis=1)[:, None])
    mx = official[(official.indicator == "DEATHRATE") & (official.age != "95+")].copy()
    mx["age"] = mx.age.astype(int)
    np.testing.assert_allclose(rates.sum(axis=1) / 100_000, mx.sort_values("age").value)


def test_old_age_curve_is_continuous_increasing_and_bounded(inputs):
    model = MortalityModel(inputs[1], official_life_table=inputs[2])
    config = ProjectionConfig()
    rates = (
        model.baseline_total(np.array([94, 94.000001, 95, 100, 130, 200, 500]), config) / 100_000
    )
    assert rates[1] == pytest.approx(rates[0], abs=1e-7)
    assert (np.diff(rates) >= 0).all()
    assert (rates <= 1).all()
    assert rates[-1] == pytest.approx(1)


def test_covid_has_separate_path_and_non_covid_is_unchanged(model):
    ages = np.array([70, 70, 70])
    years = np.array([2021, 2024, 2051])
    fade = ProjectionConfig(trend_horizon_years=0)
    hold = replace(fade, pandemic=PandemicConfig(mode="hold_2021"))
    a, b = model.rates(ages, years, fade), model.rates(ages, years, hold)
    np.testing.assert_allclose(a[:, ~model.covid_mask], b[:, ~model.covid_mask])
    np.testing.assert_allclose(
        a[:, model.covid_mask].sum(axis=1) / b[:, model.covid_mask].sum(axis=1),
        [0.1 + 0.9 * 2 ** (-t / 3) for t in (0, 3, 30)],
    )


def test_pre_pandemic_trends_do_not_learn_from_future_years(inputs):
    raw, frame, _ = inputs
    first = build_trend_model(raw, frame)
    changed = raw.copy()
    changed.loc[changed.year >= 2020, "rate"] *= 1000
    second = build_trend_model(changed, frame)
    assert first == second
    assert first.total_count == 66 * 13
    assert all(not code.startswith("U") for rates in first.slope_table.values() for code in rates)


def test_fixed_calendar_trend_horizon_is_independent_of_intervention_dates(model):
    config = ProjectionConfig(pandemic=PandemicConfig(mode="hold_2021"), trend_horizon_years=20)
    rates = model.rates(np.array([85, 85]), np.array([2041, 2100]), config)
    np.testing.assert_array_equal(rates[0], rates[1])


def test_sensitivity_current_row_reproduces_projection_and_does_not_mutate_inputs(model):
    config = ProjectionConfig()
    result = simulate_survival(model, 35, config=config)
    sensitivity = run_sensitivity(model, 35, {}, config)
    assert sensitivity.iloc[0]["Scenario mean age"] == result.life_expectancy_age
    assert sensitivity.Assumption.is_unique
    assert (sensitivity["Gain (years)"] == 0).all()
    assert config == ProjectionConfig()
