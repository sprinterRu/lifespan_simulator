"""A reusable age/cause surface, calibrated against an optional official life table."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .cause_models import EXTERNAL_CAUSES, LINKED_CAUSES
from .config import BASE_YEAR, COVID_CODES, RATE_SCALE, PandemicConfig, ProjectionConfig
from .interventions import ScenarioMap, log_intervention_multiplier
from .tails import (
    LAST_CLOSED_MIDPOINT,
    TAIL_ANCHOR_AGE,
    CauseTailFit,
    fit_cause_tails,
)

TrendTable = dict[int, dict[str, float]]


@dataclass(frozen=True)
class FutureMortality:
    minimum_hazard: float
    infinite_mean: bool
    remaining_hazard_upper_bound: float = float("inf")


@dataclass(frozen=True)
class _FutureTerm:
    index: int
    log_rate: float
    growth: float = 0.0
    upper_factor: float = 0.0
    power: float = 0.0
    power_scale: float = 1.0


def _log_nonnegative(values: np.ndarray | float) -> np.ndarray:
    """Represent zero rates/multipliers as -inf without logarithm warnings."""
    values = np.asarray(values, dtype=float)
    return np.log(values, out=np.full_like(values, -np.inf), where=values > 0)


class MortalityModel:
    """Prepare interpolation arrays once, then share them across scenarios.

    Calibration preserves interpolated cause shares and replaces their summed
    2021 rate with official single-age Mx. Direct tails splice at 94; reporting
    categories follow identified natural or external causes. Legacy common curves remain available
    for comparison. No separate 'old age' cause is added.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        trend_table: TrendTable | None = None,
        official_life_table: pd.DataFrame | None = None,
        tail_fits: dict[str, CauseTailFit] | None = None,
    ):
        required = {"cause_code", "cause_name", "age_start", "age_mid", "rate"}
        if not required.issubset(frame) or frame.empty:
            raise ValueError("Mortality frame is empty or missing required columns")
        frame = frame[frame.age_start < 95].copy()
        if frame.empty:
            raise ValueError("Mortality model requires closed age groups below 95")
        if frame.duplicated(["cause_code", "age_start"]).any():
            raise ValueError("Duplicate cause/age observations")
        if not np.isfinite(frame.rate).all() or (frame.rate < 0).any():
            raise ValueError("Mortality rates must be finite and non-negative")
        self.codes = tuple(sorted(frame.cause_code.unique()))
        self.names = (
            frame.drop_duplicates("cause_code").set_index("cause_code").cause_name.to_dict()
        )
        self.minimum_age = int(frame.age_start.min())
        self.tail_fits = (
            dict(tail_fits) if tail_fits is not None else fit_cause_tails(frame, self.codes)
        )
        if set(self.tail_fits) != set(self.codes):
            raise ValueError("Tail fits must cover exactly the model's causes")
        self.linked_donors = {}
        for index, code in enumerate(self.codes):
            curve = self.tail_fits[code].curve
            if curve.kind == "linked":
                self.linked_donors[index] = [
                    j
                    for j, donor in enumerate(self.codes)
                    if donor not in LINKED_CAUSES
                    and self.tail_fits[donor].curve.kind != "linked"
                    and (donor in EXTERNAL_CAUSES) == (curve.linked_group == "external")
                ]
        self.covid_mask = np.array([code in COVID_CODES for code in self.codes])
        self._points = []
        self._slopes = []
        for code in self.codes:
            group = frame[frame.cause_code == code].sort_values("age_start")
            ages = group.age_mid.to_numpy(dtype=float)
            self._points.append((ages, group.rate.to_numpy(dtype=float)))
            slopes = np.array(
                [(trend_table or {}).get(int(age), {}).get(code, 0.0) for age in group.age_start],
                dtype=float,
            )
            self._slopes.append(np.zeros_like(slopes) if code in COVID_CODES else slopes)
        self.official = official_life_table
        if official_life_table is not None:
            numeric = official_life_table[official_life_table.age != "95+"].copy()
            numeric["age"] = numeric.age.astype(int)
            rates = numeric[numeric.indicator == "DEATHRATE"].sort_values("age")
            if (
                rates.age.tolist() != list(range(35, 95))
                or not np.isfinite(rates.value).all()
                or (rates.value <= 0).any()
            ):
                raise ValueError("Official reference must contain positive Mx for every age 35–94")
            self._official_ages = rates.age.to_numpy()
            self._official_hazards = rates.value.to_numpy()

    def _raw_rates(self, ages: np.ndarray) -> np.ndarray:
        return np.column_stack([np.interp(ages, x, y) for x, y in self._points])

    def _trend_slopes(self, ages: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [
                np.interp(ages, points[0], slopes)
                for points, slopes in zip(self._points, self._slopes, strict=True)
            ]
        )

    def _linked_log_ratios(self, raw: np.ndarray) -> dict[int, np.ndarray]:
        """Observed reporting/identified ratios; no linked category is a donor."""
        ratios = {}
        for index, donors in self.linked_donors.items():
            denominator = raw[:, donors].sum(axis=1)
            if ((denominator == 0) & (raw[:, index] > 0)).any():
                raise ValueError(
                    f"Linked cause {self.codes[index]} has no positive reference group"
                )
            ratios[index] = _log_nonnegative(
                np.divide(raw[:, index], denominator, out=np.zeros(len(raw)), where=denominator > 0)
            )
        return ratios

    def _observed_total(self, ages: np.ndarray, calibrate: bool) -> np.ndarray:
        if calibrate:
            if self.official is None:
                raise ValueError("Official life table is required for calibration")
            return np.interp(ages, self._official_ages, self._official_hazards) * RATE_SCALE
        return self._raw_rates(ages).sum(axis=1)

    def old_age_fit(self, config: ProjectionConfig) -> dict[str, float]:
        """Fit unweighted log/logit slopes at ages 80–94, anchored at 94.

        This is a transparent approximation to an old-age fit, not the HMD's
        exposure-weighted likelihood estimator. The Kannisto asymptote is 1/year.
        """
        ages = np.arange(80, 95)
        hazards = self._observed_total(ages, config.calibrate) / RATE_SCALE
        if not np.isfinite(hazards).all() or (hazards <= 0).any():
            raise ValueError("Old-age fitting requires positive mortality at ages 80–94")
        gompertz = max(0.0, float(np.polyfit(ages, np.log(hazards), 1)[0]))
        if (hazards >= 1).any():
            raise ValueError("Kannisto fit requires observed hazards below 1 per year")
        kannisto = max(0.0, float(np.polyfit(ages, np.log(hazards / (1 - hazards)), 1)[0]))
        return {
            "anchor_hazard": float(hazards[-1]),
            "kannisto_slope": kannisto,
            "gompertz_slope": gompertz,
            "asymptote": 1.0,
        }

    def exponential_age_growth(self, config: ProjectionConfig) -> dict[str, float]:
        """Positive log-slopes of the active exponential tails, after overrides."""
        if config.old_age_model == "individual":
            return {
                code: fit.curve.slope * config.old_age_slope_multiplier
                for code, fit in self.tail_fits.items()
                if fit.curve.kind == "exponent" and fit.curve.slope > 0
            }
        if config.old_age_model == "gompertz":
            slope = self.old_age_fit(config)["gompertz_slope"] * config.old_age_slope_multiplier
            return dict.fromkeys(self.codes, slope) if slope > 0 else {}
        return {}

    def growing_causes(self, config: ProjectionConfig) -> set[str]:
        """Direct growing hazards, excluding reporting categories and flat curves."""
        if config.old_age_model == "individual":
            return {
                code
                for code, fit in self.tail_fits.items()
                if fit.curve.kind in {"power", "bounded", "exponent", "sigmoid"}
                and fit.curve.slope > 0
                and (fit.curve.kind != "sigmoid" or fit.curve.asymptote_ratio > 1)
            }
        if config.old_age_model in {"gompertz", "kannisto"}:
            return (
                set(self.codes)
                if self.old_age_fit(config)[config.old_age_model + "_slope"] > 0
                else set()
            )
        return set()

    def baseline_total(self, ages: np.ndarray, config: ProjectionConfig) -> np.ndarray:
        with np.errstate(over="ignore"):
            return np.exp(self._log_baseline_hazard(ages, config)) * RATE_SCALE

    def _log_baseline_hazard(self, ages: np.ndarray, config: ProjectionConfig) -> np.ndarray:
        """Keep exponential ageing finite until all risk multipliers are combined."""
        if config.old_age_model == "individual":
            base_rates = self._log_rates(ages, np.full(len(ages), BASE_YEAR), config)
            return np.logaddexp.reduce(base_rates, axis=1) - np.log(RATE_SCALE)
        totals = np.log(self._observed_total(np.minimum(ages, 94), config.calibrate) / RATE_SCALE)
        beyond = ages > 94
        if not beyond.any():
            return totals
        if config.old_age_model == "flat":
            return totals
        fit = self.old_age_fit(config)
        years = ages[beyond] - 94
        anchor = fit["anchor_hazard"]
        slope = fit[config.old_age_model + "_slope"] * config.old_age_slope_multiplier
        if config.old_age_model == "kannisto":
            log_hazards = -np.logaddexp(0, np.log(1 / anchor - 1) - slope * years)
        else:
            log_hazards = np.log(anchor) + slope * years
        totals[beyond] = log_hazards
        return totals

    def rates(
        self,
        ages: np.ndarray,
        years: np.ndarray,
        config: ProjectionConfig,
        scenarios: ScenarioMap | None = None,
    ) -> np.ndarray:
        """Resolve all causes together; return rates per 100,000, not probabilities."""
        with np.errstate(over="ignore"):
            return np.exp(self._log_rates(ages, years, config, scenarios))

    def _log_rates(
        self,
        ages: np.ndarray,
        years: np.ndarray,
        config: ProjectionConfig,
        scenarios: ScenarioMap | None = None,
    ) -> np.ndarray:
        ages, years = np.asarray(ages), np.asarray(years)
        individual = config.old_age_model == "individual"
        # The open 95+ bucket is excluded. Hold the last closed group's shares
        # up to the age-94 splice, then apply independent cause multipliers.
        source_ages = np.minimum(ages, LAST_CLOSED_MIDPOINT) if individual else ages
        raw = self._raw_rates(source_ages)
        total = raw.sum(axis=1)
        if (total <= 0).any():
            raise ValueError("Cause totals must be positive at every projected age")
        if individual:
            log_scale = (
                np.log(self._observed_total(np.minimum(ages, TAIL_ANCHOR_AGE), True) / total)
                if config.calibrate
                else np.zeros(len(ages))
            )
        else:
            log_scale = self._log_baseline_hazard(ages, config) - np.log(total / RATE_SCALE)
        elapsed = np.clip(years - BASE_YEAR, 0, config.trend_horizon_years)
        projected = raw + self._trend_slopes(source_ages) * elapsed[:, None]
        projected = np.maximum(raw * config.trend_floor, projected)
        # A zero observed cause rate does not impose a zero ceiling on new risk.
        projected = np.minimum(np.where(raw > 0, raw * config.trend_ceiling, np.inf), projected)
        log_rates = _log_nonnegative(projected) + log_scale[:, None]
        if individual:
            distance = np.maximum(ages - TAIL_ANCHOR_AGE, 0)
            for index, code in enumerate(self.codes):
                if index in self.linked_donors:
                    continue
                log_rates[:, index] += self.tail_fits[code].curve.log_multiplier(
                    distance, config.old_age_slope_multiplier, config.decline_residual_fraction
                )
        pandemic = config.pandemic
        if pandemic.mode == "exclude":
            log_covid_multiplier = np.full(len(years), -np.inf)
        elif pandemic.mode == "hold_2021":
            log_covid_multiplier = np.zeros(len(years))
        else:
            log_decay = -np.log(2) * np.maximum(0, years - BASE_YEAR) / pandemic.half_life_years
            log_residual, log_fading_share = _log_nonnegative(
                np.array([pandemic.residual_fraction, 1 - pandemic.residual_fraction])
            )
            log_covid_multiplier = np.logaddexp(log_residual, log_fading_share + log_decay)
        log_rates[:, self.covid_mask] += log_covid_multiplier[:, None]
        for code, intervention in (scenarios or {}).items():
            if code not in self.codes:
                raise ValueError(f"Unknown intervention cause: {code}")
            index = self.codes.index(code)
            if not individual or index not in self.linked_donors:
                log_rates[:, index] += log_intervention_multiplier(years, intervention)
        if individual:
            for index, ratio in self._linked_log_ratios(raw).items():
                donors = self.linked_donors[index]
                log_rates[:, index] = ratio + np.logaddexp.reduce(log_rates[:, donors], axis=1)
                intervention = (scenarios or {}).get(self.codes[index])
                if intervention is not None:
                    log_rates[:, index] += log_intervention_multiplier(years, intervention)
        return log_rates

    def future_hazard_lower_bound(
        self, age: int, year: int, config: ProjectionConfig, scenarios: ScenarioMap
    ) -> float:
        return self.future_mortality(age, year, config, scenarios).minimum_hazard

    def future_mortality(
        self, age: int, year: int, config: ProjectionConfig, scenarios: ScenarioMap
    ) -> FutureMortality:
        """Bound the tail after trends and intervention stage transitions finish.

        Age curves supply constant, exponential, polynomial or bounded terms;
        decline-to-residual curves split into constant and decreasing terms.
        Every remaining term decaying geometrically implies finite cumulative
        future hazard, positive limiting survival, and an infinite full mean.
        Non-decaying terms instead provide a positive future hazard bound.
        """
        if age < 95:
            raise ValueError("Tail bound requires age at least 95")
        anchor_year = max(
            year,
            BASE_YEAR + config.trend_horizon_years,
            max((value.fast_start_year for value in scenarios.values()), default=year),
        )
        anchor_age = age + anchor_year - year
        fixed_pandemic = replace(config, pandemic=PandemicConfig(mode="hold_2021"))
        log_rates = self._log_rates(
            np.array([anchor_age]),
            np.array([anchor_year]),
            fixed_pandemic,
            scenarios,
        )[0] - np.log(RATE_SCALE)
        # Split direct causes first, then propagate each donor term into linked
        # reporting categories. This carries donor treatment and COVID decay into
        # their bounds without assigning the reporting code an independent slope.
        terms = []
        for term in self._future_age_terms(log_rates, anchor_age, config):
            if not np.isfinite(term.log_rate):
                continue
            intervention = scenarios.get(self.codes[term.index])
            growth = term.growth
            if intervention is not None:
                growth += (
                    np.log1p(-intervention.fast_rate) if intervention.fast_rate < 1 else -np.inf
                )
            pandemic_terms = [(0.0, 0.0)]
            if self.covid_mask[term.index]:
                pandemic = config.pandemic
                if pandemic.mode == "exclude":
                    pandemic_terms = []
                elif pandemic.mode == "fade":
                    residual = pandemic.residual_fraction
                    decay = -np.log(2) / pandemic.half_life_years
                    pandemic_terms = []
                    if residual > 0:
                        pandemic_terms.append((np.log(residual), 0.0))
                    if residual < 1:
                        pandemic_terms.append(
                            (np.log1p(-residual) + decay * (anchor_year - BASE_YEAR), decay)
                        )
            for log_multiplier, pandemic_growth in pandemic_terms:
                terms.append(
                    replace(
                        term,
                        log_rate=term.log_rate + log_multiplier,
                        growth=growth + pandemic_growth,
                    )
                )
        if config.old_age_model == "individual":
            direct_terms = list(terms)
            raw = self._raw_rates(np.array([LAST_CLOSED_MIDPOINT]))
            for index, ratios in self._linked_log_ratios(raw).items():
                ratio = float(ratios[0])
                extra_growth = 0.0
                intervention = scenarios.get(self.codes[index])
                if intervention is not None:
                    ratio += float(
                        log_intervention_multiplier(np.array([anchor_year]), intervention)[0]
                    )
                    extra_growth = (
                        np.log1p(-intervention.fast_rate) if intervention.fast_rate < 1 else -np.inf
                    )
                if not np.isfinite(ratio):
                    continue
                for term in direct_terms:
                    if term.index in self.linked_donors[index]:
                        terms.append(
                            replace(
                                term,
                                index=index,
                                log_rate=term.log_rate + ratio,
                                growth=term.growth + extra_growth,
                            )
                        )

        prefix_min, prefix_sum = float("inf"), 0.0
        if anchor_age > age:
            prefix_ages = np.arange(age, anchor_age)
            with np.errstate(over="ignore"):
                prefix = (
                    self.rates(prefix_ages, year + prefix_ages - age, config, scenarios).sum(axis=1)
                    / RATE_SCALE
                )
                prefix_min, prefix_sum = float(prefix.min()), float(prefix.sum())
        non_decaying = [term for term in terms if term.growth >= 0]
        with np.errstate(over="ignore"):
            if non_decaying:
                lower = float(np.exp(np.logaddexp.reduce([term.log_rate for term in non_decaying])))
                return FutureMortality(min(prefix_min, lower), infinite_mean=False)
            if not terms:
                return FutureMortality(
                    0.0, infinite_mean=True, remaining_hazard_upper_bound=prefix_sum
                )
            log_sums = []
            for term in terms:
                growth, envelope = term.growth, term.upper_factor
                if term.power > 0 and np.isfinite(growth):
                    # Bound (1+n/s)^p by exp(epsilon*n+C), with epsilon
                    # small enough to leave geometric decay. Maximize the
                    # concave difference analytically over all n >= 0.
                    epsilon = min(-growth / 2, term.power / term.power_scale)
                    peak = max(0.0, term.power / epsilon - term.power_scale)
                    envelope += term.power * np.log1p(peak / term.power_scale) - epsilon * peak
                    growth += epsilon
                log_sums.append(term.log_rate + envelope - np.log(-np.expm1(growth)))
            upper = prefix_sum + float(np.exp(np.logaddexp.reduce(log_sums)))
            return FutureMortality(0.0, infinite_mean=True, remaining_hazard_upper_bound=upper)

    def _future_age_terms(
        self, log_rates: np.ndarray, anchor_age: int, config: ProjectionConfig
    ) -> list[_FutureTerm]:
        """Nonnegative direct-cause terms with geometric/polynomial envelopes."""
        if config.old_age_model != "individual":
            growth = (
                self.old_age_fit(config)["gompertz_slope"] * config.old_age_slope_multiplier
                if config.old_age_model == "gompertz"
                else 0.0
            )
            upper = (
                -float(self._log_baseline_hazard(np.array([anchor_age]), config)[0])
                if config.old_age_model == "kannisto"
                else 0.0
            )
            return [_FutureTerm(i, float(rate), growth, upper) for i, rate in enumerate(log_rates)]
        terms = []
        distance = anchor_age - TAIL_ANCHOR_AGE
        for index, code in enumerate(self.codes):
            curve = self.tail_fits[code].curve
            if curve.kind == "linked":
                continue
            rate = float(log_rates[index])
            slope = curve.slope * config.old_age_slope_multiplier
            if curve.kind == "exponent":
                terms.append(_FutureTerm(index, rate, slope))
            elif curve.kind == "decline":
                residual = config.decline_residual_fraction
                if residual == 0:
                    terms.append(_FutureTerm(index, rate, -slope))
                else:
                    base = rate - float(
                        curve.log_multiplier(
                            np.array([distance]), config.old_age_slope_multiplier, residual
                        )[0]
                    )
                    terms.append(_FutureTerm(index, base + np.log(residual)))
                    terms.append(
                        _FutureTerm(index, base + np.log1p(-residual) - slope * distance, -slope)
                    )
            elif curve.kind == "sigmoid":
                current = float(
                    curve.log_multiplier(np.array([distance]), config.old_age_slope_multiplier)[0]
                )
                terms.append(
                    _FutureTerm(
                        index, rate, upper_factor=max(0.0, np.log(curve.asymptote_ratio) - current)
                    )
                )
            elif curve.kind == "power":
                terms.append(
                    _FutureTerm(
                        index,
                        rate,
                        power=slope * curve.slowdown_years,
                        power_scale=curve.slowdown_years + distance,
                    )
                )
            elif curve.kind == "bounded":
                remaining = slope * curve.slowdown_years * np.exp(-distance / curve.slowdown_years)
                terms.append(_FutureTerm(index, rate, upper_factor=remaining))
            else:
                terms.append(_FutureTerm(index, rate))
        return terms
