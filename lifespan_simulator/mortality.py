"""A reusable age/cause surface, calibrated against an optional official life table."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .config import BASE_YEAR, COVID_CODES, RATE_SCALE, PandemicConfig, ProjectionConfig
from .interventions import ScenarioMap, intervention_multiplier, limiting_intervention_multiplier

TrendTable = dict[int, dict[str, float]]


def _log_nonnegative(values: np.ndarray | float) -> np.ndarray:
    """Represent zero rates/multipliers as -inf without logarithm warnings."""
    values = np.asarray(values, dtype=float)
    return np.log(values, out=np.full_like(values, -np.inf), where=values > 0)


class MortalityModel:
    """Prepare interpolation arrays once, then share them across scenarios.

    Calibration preserves interpolated cause shares and replaces their summed
    2021 rate with official single-age Mx. Beyond 94, total mortality follows an
    anchored old-age curve; cause shares become fixed at 95 because finer data
    are unavailable. No separate 'old age' cause is added.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        trend_table: TrendTable | None = None,
        official_life_table: pd.DataFrame | None = None,
    ):
        required = {"cause_code", "cause_name", "age_start", "age_mid", "rate"}
        if not required.issubset(frame) or frame.empty:
            raise ValueError("Mortality frame is empty or missing required columns")
        if frame.duplicated(["cause_code", "age_start"]).any():
            raise ValueError("Duplicate cause/age observations")
        if not np.isfinite(frame.rate).all() or (frame.rate < 0).any():
            raise ValueError("Mortality rates must be finite and non-negative")
        self.codes = tuple(sorted(frame.cause_code.unique()))
        self.names = (
            frame.drop_duplicates("cause_code").set_index("cause_code").cause_name.to_dict()
        )
        self.minimum_age = int(frame.age_start.min())
        self.covid_mask = np.array([code in COVID_CODES for code in self.codes])
        self._points = []
        self._slopes = []
        for code in self.codes:
            group = frame[frame.cause_code == code].sort_values("age_start")
            # Put the open bucket at its boundary; a fictitious midpoint of 97
            # followed by an override at 95 caused a discontinuity in the old code.
            ages = np.where(group.age_start == 95, 95, group.age_mid).astype(float)
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
            self._official_open_hazard = float(
                official_life_table.loc[
                    (official_life_table.age == "95+")
                    & (official_life_table.indicator == "DEATHRATE"),
                    "value",
                ].iloc[0]
            )

    def _raw_rates(self, ages: np.ndarray) -> np.ndarray:
        return np.column_stack([np.interp(ages, x, y) for x, y in self._points])

    def _trend_slopes(self, ages: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [
                np.interp(ages, points[0], slopes)
                for points, slopes in zip(self._points, self._slopes, strict=True)
            ]
        )

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

    def baseline_total(self, ages: np.ndarray, config: ProjectionConfig) -> np.ndarray:
        with np.errstate(over="ignore"):
            return np.exp(self._log_baseline_hazard(ages, config)) * RATE_SCALE

    def _log_baseline_hazard(self, ages: np.ndarray, config: ProjectionConfig) -> np.ndarray:
        """Keep exponential ageing finite until all risk multipliers are combined."""
        totals = np.log(self._observed_total(np.minimum(ages, 94), config.calibrate) / RATE_SCALE)
        beyond = ages > 94
        if not beyond.any():
            return totals
        if config.old_age_model == "flat":
            terminal = (
                self._official_open_hazard * RATE_SCALE
                if config.calibrate
                else self._raw_rates(np.array([95])).sum()
            )
            totals[beyond] = np.log(terminal / RATE_SCALE)
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
        raw = self._raw_rates(ages)
        total = raw.sum(axis=1)
        if (total <= 0).any():
            raise ValueError("Cause totals must be positive at every projected age")
        log_scale = self._log_baseline_hazard(ages, config) - np.log(total / RATE_SCALE)
        elapsed = np.clip(years - BASE_YEAR, 0, config.trend_horizon_years)
        projected = raw + self._trend_slopes(ages) * elapsed[:, None]
        projected = np.maximum(raw * config.trend_floor, projected)
        # A zero observed cause rate does not impose a zero ceiling on new risk.
        projected = np.minimum(np.where(raw > 0, raw * config.trend_ceiling, np.inf), projected)
        log_rates = _log_nonnegative(projected) + log_scale[:, None]
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
            log_rates[:, self.codes.index(code)] += _log_nonnegative(
                intervention_multiplier(years, intervention)
            )
        return log_rates

    def future_hazard_lower_bound(
        self, age: int, year: int, config: ProjectionConfig, scenarios: ScenarioMap
    ) -> float:
        """Conservative future hazard bound after shares/slopes stop changing with age.

        Old-age totals are non-decreasing. Each time trend is monotone until its
        fixed horizon, and intervention/COVID multipliers cannot fall below their
        configured floors. Thus S(age)/bound limits the unintegrated life-years.
        """
        if age < 95:
            raise ValueError("Tail bound requires age at least 95")
        fixed_pandemic = replace(config, pandemic=PandemicConfig(mode="hold_2021"))
        log_rates = self._log_rates(
            np.array([age, age]),
            np.array([year, max(year, BASE_YEAR + config.trend_horizon_years)]),
            fixed_pandemic,
        )
        log_lower = log_rates.min(axis=0)
        log_lower[self.covid_mask] += _log_nonnegative(config.pandemic.limiting_multiplier)
        for code, intervention in scenarios.items():
            log_lower[self.codes.index(code)] += _log_nonnegative(
                limiting_intervention_multiplier(intervention)
            )
        with np.errstate(over="ignore"):
            return float(np.exp(np.logaddexp.reduce(log_lower) - np.log(RATE_SCALE)))
