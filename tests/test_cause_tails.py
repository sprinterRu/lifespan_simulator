from dataclasses import replace
from math import exp, log

import numpy as np
import pandas as pd
import pytest

from lifespan_simulator.cause_models import CAUSE_ASSUMPTIONS, LINKED_CAUSES
from lifespan_simulator.config import PandemicConfig, ProjectionConfig
from lifespan_simulator.data import (
    DEMENTIA_CODE,
    DEMENTIA_COMPONENTS,
    get_simulator_frame,
    merge_dementia,
)
from lifespan_simulator.interventions import build_scenario_map, default_fast_rate
from lifespan_simulator.mortality import MortalityModel
from lifespan_simulator.simulation import simulate_survival
from lifespan_simulator.tails import CauseTailFit, TailCurve, fit_cause_tails
from lifespan_simulator.trends import build_trend_model


def one_cause_model(curve, code="X"):
    frame = pd.DataFrame(
        [
            {
                "cause_code": code,
                "cause_name": code,
                "age_start": age,
                "age_mid": age + 2,
                "rate": 1000.0,
            }
            for age in range(35, 100, 5)
        ]
    )
    fit = CauseTailFit(curve, "synthetic", "synthetic", "Test", "Known analytical curve")
    return MortalityModel(frame, tail_fits={code: fit})


def individual_config(**kwargs):
    return ProjectionConfig(
        calibrate=False,
        old_age_model="individual",
        trend_horizon_years=0,
        pandemic=PandemicConfig(mode="hold_2021"),
        **kwargs,
    )


def test_merge_preserves_every_year_age_total_and_precedes_time_fits(inputs):
    raw, frame, _ = inputs
    merged = merge_dementia(raw)
    assert not set(merged.cause_code) & DEMENTIA_COMPONENTS
    assert frame.cause_code.nunique() == 65
    np.testing.assert_allclose(
        merged.groupby(["year", "age_start"]).rate.sum(),
        raw.groupby(["year", "age_start"]).rate.sum(),
    )
    pd.testing.assert_frame_equal(merge_dementia(merged), merged)
    expected = (
        raw[raw.cause_code.isin(DEMENTIA_COMPONENTS)].groupby(["year", "age_start"]).rate.sum()
    )
    actual = merged[merged.cause_code == DEMENTIA_CODE].set_index(["year", "age_start"]).rate
    np.testing.assert_allclose(actual.sort_index(), expected.sort_index())
    trends = build_trend_model(raw, frame)
    assert all(DEMENTIA_CODE in bucket for bucket in trends.slope_table.values())
    assert all(not DEMENTIA_COMPONENTS & bucket.keys() for bucket in trends.slope_table.values())


def test_merge_rejects_missing_component(inputs):
    raw = inputs[0]
    with pytest.raises(ValueError, match="both component"):
        merge_dementia(raw[raw.cause_code != "G30"])


def test_shape_selection_does_not_use_pandemic_or_open_age_rates(inputs, model):
    raw, _, _ = inputs
    changed = raw.copy()
    changed.loc[(changed.year >= 2020) | (changed.age_start == 95), "rate"] *= 1000
    assert fit_cause_tails(changed, model.codes) == model.tail_fits
    assert model.tail_fits["C33_C34"].curve.kind == "decline"
    assert model.tail_fits["C33_C34"].validation_log_rmse is None
    assert model.tail_fits["G20"].curve.kind == "power"
    assert model.tail_fits["B20-B24"].status == "Provisional assumption"
    assert model.tail_fits["U071"].status == "Provisional assumption"


def test_reviewed_dementia_uses_nearby_slope_and_preserves_comparison(inputs, model):
    fit = model.tail_fits[DEMENTIA_CODE]
    assert fit.curve.kind == "power"
    assert fit.fit_ages == "80–94"
    assert fit.curve.slope == pytest.approx(0.1497565669)
    assert fit.curve.slowdown_years == 20
    assert fit.status == "Scenario assumption"
    assert fit.validation_log_rmse is None and fit.year_agreement is None
    assert fit.alternative_curve.kind == "sigmoid"
    assert fit.alternative_curve.slope == pytest.approx(0.2167405741)
    assert fit.alternative_curve.asymptote_ratio == pytest.approx(1.528725466)
    assert fit.alternative_log_rmse == pytest.approx(0.1165226056)
    raw = inputs[0]
    changed = raw.copy()
    changed.loc[changed.age_start == 90, "rate"] *= 2
    assert fit_cause_tails(changed, [DEMENTIA_CODE])[DEMENTIA_CODE].curve.slope > fit.curve.slope
    assert fit.record()["description"] and fit.record()["rationale"]


@pytest.mark.parametrize(
    "curve",
    [
        TailCurve("exponent", 0.12),
        TailCurve("sigmoid", 0.2, 4),
        TailCurve(),
    ],
)
def test_simple_synthetic_profiles_choose_their_family(curve):
    rows = []
    for year in range(2011, 2020):
        for age in range(70, 95, 5):
            rate = 100 * np.exp(curve.log_multiplier(np.array([age + 2 - 72]))[0])
            rows.append(
                {
                    "year": year,
                    "age_start": age,
                    "age_mid": age + 2,
                    "rate": rate * (1 + 0.01 * (year - 2011)),
                    "cause_code": "X",
                }
            )
    result = fit_cause_tails(pd.DataFrame(rows), ["X"])["X"]
    assert result.curve.kind == curve.kind
    assert result.validation_log_rmse is not None


@pytest.mark.parametrize(
    "curve",
    [
        TailCurve("exponent", 0.2),
        TailCurve("sigmoid", 0.2, 3),
        TailCurve(),
        TailCurve("decline", 0.1),
        TailCurve("power", 0.15),
        TailCurve("bounded", 0.15),
    ],
)
def test_individual_curves_splice_continuously_and_follow_their_shape(curve):
    model = one_cause_model(curve)
    ages = np.array([94, 94.000001, 95, 120, 300])
    actual = model.rates(ages, np.full(len(ages), 2021), individual_config())[:, 0]
    assert actual[1] == pytest.approx(actual[0], rel=1e-6)
    expected = 1000 * np.exp(curve.log_multiplier(ages - 94, decline_residual=0.25))
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("mode", ["individual", "flat", "gompertz", "kannisto"])
def test_all_age_models_ignore_open_bucket_as_point_data(inputs, model, mode):
    _, frame, official = inputs
    added = frame[frame.age_start == 90].assign(age_start=95, age_mid=97, rate=1e9)
    changed = pd.concat([frame, added], ignore_index=True)
    extra_reference = official[official.age == "94"].assign(age="95+", value=1e9)
    changed_official = pd.concat([official, extra_reference], ignore_index=True)
    assert len(changed) > len(frame)
    alternative = MortalityModel(
        changed, official_life_table=changed_official, tail_fits=model.tail_fits
    )
    original = MortalityModel(frame, official_life_table=official, tail_fits=model.tail_fits)
    ages = np.arange(35, 201)
    np.testing.assert_allclose(
        alternative.rates(ages, np.full(len(ages), 2021), ProjectionConfig(old_age_model=mode)),
        original.rates(ages, np.full(len(ages), 2021), ProjectionConfig(old_age_model=mode)),
    )


def test_preparation_ignores_even_incomplete_open_dementia_pairs(inputs, model):
    raw, frame, official = inputs
    changed = raw[~((raw.age_start == 95) & (raw.cause_code == "G30"))].copy()
    changed.loc[changed.age_start == 95, "rate"] *= 1000
    pd.testing.assert_frame_equal(get_simulator_frame(changed), frame)
    assert fit_cause_tails(changed, model.codes) == model.tail_fits
    assert build_trend_model(changed, frame) == build_trend_model(raw, frame)
    assert frame.age_start.max() == 90
    assert "95+" not in set(official.age)


def test_treatment_does_not_redistribute_removed_hazard_to_other_causes(model):
    ages = np.arange(94, 151)
    years = 2021 + ages - 35
    base = model.rates(ages, years, ProjectionConfig())
    scenarios = build_scenario_map({DEMENTIA_CODE: (2021, 0, 2021, 1, 0.01)})
    treated = model.rates(ages, years, ProjectionConfig(), scenarios)
    index = model.codes.index(DEMENTIA_CODE)
    assert (treated[:, index] == 0).all()
    independent = [
        i
        for i, code in enumerate(model.codes)
        if code != DEMENTIA_CODE and code not in LINKED_CAUSES
    ]
    np.testing.assert_array_equal(treated[:, independent], base[:, independent])
    for code in ("R96-R99", "R_OTH"):
        assert (treated[:, model.codes.index(code)] < base[:, model.codes.index(code)]).all()
    assert (treated <= base).all()


def test_decline_without_treatment_has_known_finite_cumulative_hazard():
    model = one_cause_model(TailCurve("decline", 0.1))
    result = simulate_survival(
        model, 94, config=individual_config(max_age=200, decline_residual_fraction=0)
    )
    cumulative = 0.01 / (1 - exp(-0.1))
    assert result.infinite_mean
    assert result.survival_forever_lower_bound == pytest.approx(exp(-cumulative))
    assert result.percentiles["p50"] == float("inf")


def test_positive_decline_residual_restores_finite_mean_and_valid_bound():
    model = one_cause_model(TailCurve("decline", 0.1))
    config = individual_config(decline_residual_fraction=0.25, max_age=10_000)
    result = simulate_survival(model, 94, config=config)
    assert result.converged and not result.infinite_mean
    bound = model.future_mortality(120, 2047, config, {})
    assert bound.minimum_hazard == pytest.approx(0.0025)
    short = simulate_survival(model, 94, config=replace(config, max_age=120))
    assert short.restricted_remaining_life_expectancy < result.remaining_life_expectancy
    assert result.remaining_life_expectancy <= (
        short.restricted_remaining_life_expectancy + short.remaining_expectation_bound
    )


@pytest.mark.parametrize(
    "curve",
    [
        TailCurve("exponent", 0.07),
        TailCurve("sigmoid", 0.2, 10),
        TailCurve(),
        TailCurve("decline", 0.1),
        TailCurve("power", 0.15),
        TailCurve("bounded", 0.15),
    ],
)
@pytest.mark.parametrize("residual", [0.0, 0.25])
def test_decaying_future_upper_bound_covers_direct_hazard_sum(curve, residual):
    model = one_cause_model(curve)
    config = individual_config(decline_residual_fraction=residual)
    scenario = build_scenario_map({"X": (2021, 0, 2021, 0.2, 0.5)})
    future = model.future_mortality(100, 2027, config, scenario)
    ages = np.arange(100, 5001)
    actual = model.rates(ages, 2027 + ages - 100, config, scenario)[:, 0].sum() / 100_000
    assert future.infinite_mean
    assert actual <= future.remaining_hazard_upper_bound * (1 + 1e-12)
    assert future.remaining_hazard_upper_bound < float("inf")


def test_increasing_cause_can_outpace_fast_treatment():
    scenario = build_scenario_map({"X": (2021, 0, 2021, 0.1, 0.01)})
    slow = one_cause_model(TailCurve("exponent", 0.05))
    fast = one_cause_model(TailCurve("exponent", 0.2))
    assert slow.future_mortality(100, 2027, individual_config(), scenario).infinite_mean
    assert not fast.future_mortality(100, 2027, individual_config(), scenario).infinite_mean


@pytest.mark.parametrize("mode", ["individual", "gompertz"])
def test_adjusted_fast_defaults_reduce_exponential_hazards_by_ten_percent(model, mode):
    config = ProjectionConfig(
        old_age_model=mode,
        old_age_slope_multiplier=1.25,
        trend_horizon_years=0,
        pandemic=PandemicConfig(mode="hold_2021"),
    )
    if mode == "individual":
        # Exponentials remain available as explicit overrides, not defaults.
        model = one_cause_model(TailCurve("exponent", 0.15))
        config = replace(config, calibrate=False)
    growth = model.exponential_age_growth(config)
    assert growth
    scenarios = build_scenario_map(
        {
            code: (2040, 0.015, 2080, default_fast_rate(slope), 0.01)
            for code, slope in growth.items()
        }
    )
    indices = [model.codes.index(code) for code in growth]
    rates = model.rates(np.array([100, 101]), np.array([2086, 2087]), config, scenarios)[:, indices]
    np.testing.assert_allclose(rates[1], 0.9 * rates[0], rtol=1e-12)


def test_exponential_selection_uses_active_shapes_and_excludes_zero_slopes():
    for curve, selected in [
        (TailCurve("exponent", 0.2), True),
        (TailCurve("exponent", 0), False),
        (TailCurve("sigmoid", 0.2, 4), False),
        (TailCurve("decline", 0.2), False),
        (TailCurve(), False),
    ]:
        model = one_cause_model(curve)
        assert bool(model.exponential_age_growth(individual_config())) == selected
    for mode in ("flat", "kannisto"):
        assert not model.exponential_age_growth(replace(individual_config(), old_age_model=mode))


def test_adjusted_defaults_on_all_causes_outpace_every_exponential_tail(model):
    config = ProjectionConfig()
    growth = model.exponential_age_growth(config)
    scenarios = build_scenario_map(
        {
            code: (2040, 0.015, 2080, default_fast_rate(growth.get(code, 0)), 0.01)
            for code in model.codes
        }
    )
    assert default_fast_rate(0) == pytest.approx(0.1)
    assert model.future_mortality(100, 2086, config, scenarios).infinite_mean


def test_covid_fade_and_decline_residual_split_into_valid_future_terms():
    model = one_cause_model(TailCurve("decline", 0.1), code="U071")
    config = replace(
        individual_config(decline_residual_fraction=0.25),
        pandemic=PandemicConfig(residual_fraction=0.1),
    )
    future = model.future_mortality(100, 2027, config, {})
    assert not future.infinite_mean
    assert future.minimum_hazard == pytest.approx(0.01 * 0.25 * 0.1)
    scenario = build_scenario_map({"U071": (2021, 0, 2021, 0.1, 0.5)})
    future = model.future_mortality(100, 2027, config, scenario)
    ages = np.arange(100, 5001)
    actual = model.rates(ages, 2027 + ages - 100, config, scenario)[:, 0].sum() / 100_000
    assert future.infinite_mean
    assert future.remaining_hazard_upper_bound == pytest.approx(actual)


def test_remote_age_computes_small_improvements_before_exponentiating():
    model = one_cause_model(TailCurve("exponent", 0.25))
    reduction = 1 - exp(-0.24)
    scenario = build_scenario_map({"X": (2021, 0, 2021, reduction, 0.01)})
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        actual = model.rates(np.array([10000]), np.array([11986]), individual_config(), scenario)[
            0, 0
        ]
    expected = exp(log(1000) + 0.25 * (10000 - 94) + log(1 - reduction) * (11986 - 2021 + 1))
    assert actual == pytest.approx(expected)


def test_all_cause_fast_elimination_is_zero_with_individual_tails(model):
    scenario = build_scenario_map({c: (2021, 0, 2021, 1, 0.01) for c in model.codes})
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = simulate_survival(model, 35, scenario, ProjectionConfig(max_age=10000))
    assert result.infinite_mean and result.survival_forever_lower_bound == 1
    assert (result.annual_projection.hazard == 0).all()


def test_registry_covers_all_retained_causes_and_no_zero_only_cancers(inputs, model):
    from collections import Counter

    from lifespan_simulator.data import INAPPLICABLE_MALE_CAUSES

    raw, frame, _ = inputs
    assert set(model.codes) == set(CAUSE_ASSUMPTIONS)
    assert not set(model.codes) & INAPPLICABLE_MALE_CAUSES.keys()
    assert raw.loc[raw.cause_code.isin(INAPPLICABLE_MALE_CAUSES), "rate"].eq(0).all()
    assert {"C50", "P", "Q"} <= set(model.codes)
    assert Counter(f.curve.kind for f in model.tail_fits.values()) == {
        "power": 25,
        "bounded": 24,
        "plateau": 10,
        "decline": 3,
        "linked": 3,
    }
    assert all(f.description and f.rationale for f in model.tail_fits.values())
    assert all(f.validation_log_rmse is None for f in model.tail_fits.values())


def test_covid_borrowing_and_sparse_shrinkage_are_independent_of_requested_subset(inputs, model):
    raw = inputs[0]
    codes = ["U071", "U072", "U_COV19_OTH", "C50"]
    subset = fit_cause_tails(raw, codes)
    assert all(subset[code] == model.tail_fits[code] for code in codes)
    assert len({subset[code].curve for code in codes[:3]}) == 1
    assert subset["C50"].curve.slope > 0
    assert "25% own" in subset["C50"].reason


def test_new_curve_derivatives_limits_and_parameter_validation():
    b, tau = 0.15, 20.0
    power = TailCurve("power", b, slowdown_years=tau)
    bounded = TailCurve("bounded", b, slowdown_years=tau)
    for curve in (power, bounded):
        assert curve.log_multiplier(np.array([0.0]))[0] == 0
        assert curve.log_multiplier(np.array([1e-6]))[0] / 1e-6 == pytest.approx(b)
    assert power.log_multiplier(np.array([1e10]))[0] > 50
    assert bounded.log_multiplier(np.array([1e10]))[0] == pytest.approx(b * tau)
    for tau in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            TailCurve("power", b, slowdown_years=tau)
    with pytest.raises(ValueError):
        TailCurve("linked")
    with pytest.raises(ValueError, match="underlying"):
        TailCurve("linked", linked_group="natural").log_multiplier(np.array([1]))


def linked_model():
    specs = {
        "U071": (1000.0, TailCurve("power", 0.15)),
        "W00-W19": (200.0, TailCurve("bounded", 0.1)),
        "R96-R99": (100.0, TailCurve("linked", linked_group="natural")),
        "R_OTH": (50.0, TailCurve("linked", linked_group="natural")),
        "Y10-Y34_Y872": (20.0, TailCurve("linked", linked_group="external")),
    }
    frame = pd.DataFrame(
        [
            {
                "cause_code": code,
                "cause_name": code,
                "age_start": age,
                "age_mid": age + 2,
                "rate": rate,
            }
            for code, (rate, curve) in specs.items()
            for age in range(35, 95, 5)
        ]
    )
    return MortalityModel(
        frame,
        tail_fits={
            code: CauseTailFit(curve, "test", "test", "test", "test")
            for code, (_, curve) in specs.items()
        },
    )


def test_linked_categories_preserve_baseline_and_follow_correct_donors():
    model = linked_model()
    config = individual_config()
    ages = np.array([70.0, 94.0, 94.000001, 150.0, 200.0])
    years = np.full(len(ages), 2021)
    base = model.rates(ages, years, config)
    assert base[0].sum() == pytest.approx(1370.0)
    scenario = build_scenario_map({"U071": (2021, 0, 2021, 0.5, 0.01)})
    changed = model.rates(ages, years, config, scenario)
    for code in ("U071", "R96-R99", "R_OTH"):
        idx = model.codes.index(code)
        np.testing.assert_allclose(changed[:, idx], 0.5 * base[:, idx])
    for code in ("W00-W19", "Y10-Y34_Y872"):
        idx = model.codes.index(code)
        np.testing.assert_allclose(changed[:, idx], base[:, idx])
    assert base[2].sum() == pytest.approx(base[1].sum(), rel=1e-6)
    # Applying a reporting-category intervention is an explicit extra multiplier.
    scenario["R96-R99"] = build_scenario_map({"R96-R99": (2021, 0, 2021, 0.2, 0.01)})["R96-R99"]
    extra = model.rates(ages, years, config, scenario)
    idx = model.codes.index("R96-R99")
    np.testing.assert_allclose(extra[:, idx], 0.8 * changed[:, idx])


@pytest.mark.parametrize("fast", [0.00001, 0.02, 0.2])
def test_polynomial_future_bound_handles_growth_before_eventual_treatment_decay(fast):
    model = one_cause_model(TailCurve("power", 0.25, slowdown_years=30))
    config = individual_config()
    scenario = build_scenario_map({"X": (2021, 0, 2021, fast, 0.01)})
    future = model.future_mortality(95, 2022, config, scenario)
    ages = np.arange(95, 10001)
    partial = model.rates(ages, 2022 + ages - 95, config, scenario).sum() / 100_000
    assert future.infinite_mean
    assert partial <= future.remaining_hazard_upper_bound * (1 + 1e-12)
    assert model.future_mortality(95, 2022, config, {}).minimum_hazard > 0


def test_linked_future_bounds_include_donor_covid_and_treatment_effects():
    model = linked_model()
    config = replace(individual_config(), pandemic=PandemicConfig(residual_fraction=0.1))
    scenario = build_scenario_map(
        {code: (2021, 0, 2021, 0.2, 0.01) for code in ("U071", "W00-W19")}
    )
    future = model.future_mortality(100, 2027, config, scenario)
    ages = np.arange(100, 10001)
    partial = model.rates(ages, 2027 + ages - 100, config, scenario).sum() / 100_000
    assert future.infinite_mean
    assert partial <= future.remaining_hazard_upper_bound * (1 + 1e-12)
    assert future.remaining_hazard_upper_bound < float("inf")
    untreated = model.future_mortality(100, 2027, config, {})
    rates = model.rates(ages, 2027 + ages - 100, config).sum(axis=1) / 100_000
    assert 0 < untreated.minimum_hazard <= rates.min()
