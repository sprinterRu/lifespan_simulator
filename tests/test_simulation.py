from dataclasses import replace
from math import exp, log

import numpy as np
import pytest

from lifespan_simulator.config import PandemicConfig, ProjectionConfig
from lifespan_simulator.interventions import build_scenario_map, intervention_multiplier
from lifespan_simulator.simulation import (
    project_cause_rates,
    project_total_rates,
    simulate_survival,
)


def test_constant_hazard_matches_analytic_integrals_and_quantiles(constant_model, fixed_config):
    result = simulate_survival(constant_model, 35, config=fixed_config)
    expected_restricted = (1 - exp(-0.1 * 165)) / 0.1
    assert result.restricted_remaining_life_expectancy == pytest.approx(expected_restricted)
    assert result.remaining_life_expectancy == pytest.approx(
        10, abs=fixed_config.expectation_tolerance
    )
    assert result.percentiles["p50"] == pytest.approx(35 + log(2) / 0.1)
    assert result.percentiles["p99"] == pytest.approx(35 + log(100) / 0.1)
    assert result.remaining_expectation_bound == pytest.approx(exp(-0.1 * 165) / 0.1)


def test_neutral_scenario_is_exactly_neutral(model):
    neutral = build_scenario_map({model.codes[0]: (2040, 0, 2080, 0, 0.01)})
    a = simulate_survival(model, 35)
    b = simulate_survival(model, 35, neutral)
    assert a.life_expectancy_age == b.life_expectancy_age
    np.testing.assert_array_equal(a.survival_curve.survival, b.survival_curve.survival)


def test_neutral_intervention_does_not_make_tail_bound_inconclusive(constant_model, fixed_config):
    scenario = build_scenario_map({"X": (2040, 0, 2080, 0, 1e-12)})
    baseline = simulate_survival(constant_model, 35, config=fixed_config)
    neutral = simulate_survival(constant_model, 35, scenario, fixed_config)
    assert neutral.remaining_expectation_bound == baseline.remaining_expectation_bound
    assert neutral.life_expectancy_age == baseline.life_expectancy_age


def test_reducing_hazards_improves_survival_and_conserves_probability(model):
    baseline = simulate_survival(model, 35)
    scenarios = build_scenario_map({"C33_C34": (2021, 0.03, 2040, 0.1, 0.01)})
    result = simulate_survival(model, 35, scenarios)
    assert result.life_expectancy_age >= baseline.life_expectancy_age
    assert (result.survival_curve.survival >= baseline.survival_curve.survival - 1e-14).all()
    s = result.survival_curve.survival
    assert s.between(0, 1).all() and (np.diff(s) <= 0).all()
    annual = result.annual_projection
    assert (
        annual.survival_start - annual.survival_end
    ).sum() + result.survival_at_end == pytest.approx(1)


def test_total_and_cause_charts_use_the_same_hazards_as_survival(model):
    settings = ProjectionConfig(start_year=2026, old_age_model="gompertz")
    scenario = build_scenario_map({"C33_C34": (2040, 0.015, 2080, 0.1, 0.01)})
    causes = project_cause_rates(model, 35, list(model.codes), scenario, 120, settings)
    total = project_total_rates(model, 35, scenario, 120, settings)
    result = simulate_survival(model, 35, scenario, settings)
    np.testing.assert_allclose(causes.groupby("age").scenario_rate.sum(), total.scenario_rate)
    np.testing.assert_allclose(
        total.scenario_rate, result.annual_projection.total_rate.iloc[: len(total)]
    )
    assert total.calendar_year.iloc[0] == 2026


def test_extreme_scenario_reports_unresolved_mean_and_keeps_entire_curve(model):
    scenario = build_scenario_map({code: (2040, 0.015, 2080, 0.1, 0.01) for code in model.codes})
    result = simulate_survival(model, 35, scenario)
    assert not result.converged and result.life_expectancy_age is None
    assert result.percentiles["p50"] is not None and result.percentiles["p99"] is None
    assert result.survival_curve.age.iloc[-1] == 200
    assert result.survival_curve.survival.iloc[-1] == result.survival_at_end
    assert result.remaining_expectation_bound > 0.001
    longer = simulate_survival(model, 35, scenario, ProjectionConfig(max_age=500))
    extra = (
        longer.restricted_remaining_life_expectancy - result.restricted_remaining_life_expectancy
    )
    assert 0 < extra < result.remaining_expectation_bound


@pytest.mark.parametrize("rate", [1e-20, 0.00001, 1.0])
def test_extreme_reduction_rates_have_bounded_work(constant_model, fixed_config, rate):
    scenarios = build_scenario_map({"X": (2021, rate, 2080, rate, 1e-12)})
    result = simulate_survival(constant_model, 35, scenarios, fixed_config)
    assert len(result.annual_projection) == 165
    assert np.isfinite(result.restricted_remaining_life_expectancy)


def test_intervention_stage_boundaries():
    config = build_scenario_map({"X": (2040, 0.1, 2042, 0.2, 0.01)})["X"]
    np.testing.assert_allclose(
        intervention_multiplier(np.arange(2039, 2044), config), [1, 0.9, 0.81, 0.648, 0.5184]
    )
    config = config._replace(slow_rate=1)
    np.testing.assert_allclose(
        intervention_multiplier(np.array([2039, 2040, 2042]), config), [1, 0.01, 0.01]
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_age": 1000},
        {"start_year": 2020},
        {"trend_floor": float("nan")},
        {"trend_ceiling": float("inf")},
        {"old_age_model": "unknown"},
    ],
)
def test_invalid_model_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ProjectionConfig(**kwargs)


def test_unknown_cause_rejected(model):
    with pytest.raises(ValueError, match="Unknown"):
        simulate_survival(model, 35, build_scenario_map({"UNKNOWN": (2040, 0.01, 2080, 0.1, 0.01)}))


def test_future_bound_is_valid_for_each_supported_age_model(model):
    for mode in ("kannisto", "gompertz", "flat"):
        config = ProjectionConfig(
            old_age_model=mode, trend_floor=0, pandemic=PandemicConfig(residual_fraction=0)
        )
        scenario = build_scenario_map({"C33_C34": (2120, 0.01, 2120, 0.1, 0.01)})
        bound = model.future_hazard_lower_bound(100, 2086, config, scenario)
        ages = np.arange(100, 501)
        actual = model.rates(ages, 2086 + ages - 100, config, scenario).sum(axis=1) / 100_000
        assert (actual >= bound - 1e-12).all()


def test_omitted_integral_is_bounded_with_low_hazards(constant_model, fixed_config):
    scenario = build_scenario_map({"X": (2021, 1, 2021, 1, 0.01)})
    result = simulate_survival(constant_model, 35, scenario, replace(fixed_config, max_age=120))
    exact_remaining = 1 / 0.001
    assert not result.converged
    assert (
        result.restricted_remaining_life_expectancy + result.remaining_expectation_bound
        == pytest.approx(exact_remaining)
    )
