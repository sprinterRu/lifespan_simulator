"""Explicit assumptions shared by projection, validation and sensitivity runs."""

from dataclasses import dataclass, field
from math import isfinite
from typing import Literal

BASE_YEAR = 2021
RATE_SCALE = 100_000.0
COVID_CODES = frozenset({"U071", "U072", "U_COV19_OTH"})
TREND_FLOOR_FRACTION_DEFAULT = 0.15
TREND_CEILING_MULTIPLIER_DEFAULT = 2.0
DEFAULT_INTEGRATION_END_AGE = 2_000
MAX_INTEGRATION_END_AGE = 10_000


@dataclass(frozen=True)
class PandemicConfig:
    """An assumed COVID path, separate from ordinary mortality trends.

    Fade retains the observed 2021 rate at t=0 and approaches a residual share
    exponentially. These parameters are scenario assumptions, not fitted forecasts.
    """

    mode: Literal["fade", "hold_2021", "exclude"] = "fade"
    half_life_years: float = 3.0
    residual_fraction: float = 0.10

    def __post_init__(self) -> None:
        if self.mode not in {"fade", "hold_2021", "exclude"}:
            raise ValueError("Unknown pandemic mode")
        if not isfinite(self.half_life_years) or self.half_life_years <= 0:
            raise ValueError("COVID half-life must be finite and positive")
        if not isfinite(self.residual_fraction) or not 0 <= self.residual_fraction <= 1:
            raise ValueError("COVID residual must be between zero and one")

    @property
    def limiting_multiplier(self) -> float:
        return (
            1.0
            if self.mode == "hold_2021"
            else 0.0
            if self.mode == "exclude"
            else self.residual_fraction
        )


@dataclass(frozen=True)
class ProjectionConfig:
    start_year: int = BASE_YEAR
    calibrate: bool = True
    old_age_model: Literal["individual", "kannisto", "gompertz", "flat"] = "individual"
    old_age_slope_multiplier: float = 1.0
    decline_residual_fraction: float = 0.0
    pandemic: PandemicConfig = field(default_factory=PandemicConfig)
    trend_floor: float = TREND_FLOOR_FRACTION_DEFAULT
    trend_ceiling: float = TREND_CEILING_MULTIPLIER_DEFAULT
    trend_horizon_years: int = 50
    max_age: int = DEFAULT_INTEGRATION_END_AGE
    expectation_tolerance: float = 0.001

    def __post_init__(self) -> None:
        if not isinstance(self.start_year, int) or not BASE_YEAR <= self.start_year <= 2120:
            raise ValueError("Start year must be an integer between 2021 and 2120")
        if self.old_age_model not in {"individual", "kannisto", "gompertz", "flat"}:
            raise ValueError("Unknown old-age model")
        if (
            not isfinite(self.decline_residual_fraction)
            or not 0 <= self.decline_residual_fraction < 1
        ):
            raise ValueError("Declining-tail residual must be finite and in [0, 1)")
        if (
            not isfinite(self.old_age_slope_multiplier)
            or not 0.25 <= self.old_age_slope_multiplier <= 2
        ):
            raise ValueError("Old-age slope multiplier must be between 0.25 and 2")
        if not isfinite(self.trend_floor) or not 0 <= self.trend_floor <= 1:
            raise ValueError("Trend floor must be between zero and one")
        if not isfinite(self.trend_ceiling) or self.trend_ceiling < 1:
            raise ValueError("Trend ceiling must be finite and at least one")
        if (
            not isinstance(self.trend_horizon_years, int)
            or not 0 <= self.trend_horizon_years <= 100
        ):
            raise ValueError("Trend horizon must be an integer from zero to 100 years")
        if not isinstance(self.max_age, int) or not 100 <= self.max_age <= MAX_INTEGRATION_END_AGE:
            raise ValueError(
                f"Integration end age must be an integer between 100 and {MAX_INTEGRATION_END_AGE}"
            )
        if not isfinite(self.expectation_tolerance) or not 0 < self.expectation_tolerance <= 0.01:
            raise ValueError("Expectation tolerance must be positive and at most 0.01 years")
