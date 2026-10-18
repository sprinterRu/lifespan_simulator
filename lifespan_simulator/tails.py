"""Small, auditable families for cause-specific mortality after age 94.

Shape selection uses pre-pandemic *rates*, not death counts or exposures. Scores
are blocked-age prediction errors, not likelihoods or confidence intervals.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .config import COVID_CODES
from .data import DEMENTIA_CODE, merge_dementia

TAIL_ANCHOR_AGE = 94
LAST_CLOSED_MIDPOINT = 92
TailKind = Literal["exponent", "sigmoid", "plateau", "decline"]
# Fixed search bounds keep the logistic fit reproducible and prevent extreme
# asymptotes from being presented as measured quantities.
SIGMOID_SLOPES = np.geomspace(0.02, 0.4, 45)
SIGMOID_RATIOS = np.geomspace(1.02, 50, 55)
MAX_SLOPE = 0.4


@dataclass(frozen=True)
class TailCurve:
    kind: TailKind = "plateau"
    slope: float = 0.0
    asymptote_ratio: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in {"exponent", "sigmoid", "plateau", "decline"}:
            raise ValueError("Unknown cause-tail family")
        if not np.isfinite(self.slope) or self.slope < 0:
            raise ValueError("Tail slope must be finite and non-negative")
        if not np.isfinite(self.asymptote_ratio) or self.asymptote_ratio < 1:
            raise ValueError("Increasing sigmoid asymptote must be at least its anchor")

    def log_multiplier(
        self, elapsed: np.ndarray, slope_multiplier: float = 1, decline_residual: float = 0
    ) -> np.ndarray:
        """Normalized to exactly one at the splice; safe at remote ages."""
        distance = np.asarray(elapsed, dtype=float)
        growth = self.slope * slope_multiplier * distance
        if self.kind == "exponent":
            return growth
        if self.kind == "decline":
            if decline_residual == 0:
                return -growth
            return np.logaddexp(np.log(decline_residual), np.log1p(-decline_residual) - growth)
        if self.kind == "sigmoid" and self.asymptote_ratio > 1:
            return np.log(self.asymptote_ratio) - np.logaddexp(
                0, np.log(self.asymptote_ratio - 1) - growth
            )
        return np.zeros_like(distance)


@dataclass(frozen=True)
class CauseTailFit:
    curve: TailCurve
    fit_ages: str
    history_years: str
    status: str
    reason: str
    validation_log_rmse: float | None = None
    alternative: str | None = None
    alternative_log_rmse: float | None = None
    year_agreement: float | None = None
    alternative_curve: TailCurve | None = None
    alternative_fit_ages: str | None = None

    def record(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class _ProfileFit:
    """A fitted log-rate profile, with a level estimated independently of shape."""

    curve: TailCurve
    reference_age: float
    log_level: float

    def predict_log(self, ages: np.ndarray) -> np.ndarray:
        return self.log_level + self.curve.log_multiplier(ages - self.reference_age)

    def at_tail_anchor(self) -> TailCurve:
        curve = self.curve
        if curve.kind == "sigmoid":
            ratio = 1 + (curve.asymptote_ratio - 1) * np.exp(
                -curve.slope * (TAIL_ANCHOR_AGE - self.reference_age)
            )
            return TailCurve("sigmoid", curve.slope, float(ratio))
        return curve


def _fit_profile(ages: np.ndarray, rates: np.ndarray, kind: TailKind) -> _ProfileFit:
    reference = float(ages[-1])
    x, y = ages - reference, np.log(rates)
    if kind == "plateau":
        return _ProfileFit(TailCurve(), reference, float(y.mean()))
    if kind in {"exponent", "decline"}:
        slope, _ = np.polyfit(x, y, 1)
        slope = (
            float(np.clip(slope, 0, MAX_SLOPE))
            if kind == "exponent"
            else float(np.clip(-slope, 0, MAX_SLOPE))
        )
        curve = TailCurve(kind, slope)
        level = float(np.mean(y - curve.log_multiplier(x)))
        return _ProfileFit(curve, reference, level)

    slopes, ratios = np.meshgrid(SIGMOID_SLOPES, SIGMOID_RATIOS, indexing="ij")
    slopes, ratios = slopes.ravel(), ratios.ravel()
    shapes = np.log(ratios[:, None]) - np.logaddexp(
        0, np.log(ratios[:, None] - 1) - slopes[:, None] * x
    )
    levels = (y - shapes).mean(axis=1)
    errors = np.mean((y - shapes - levels[:, None]) ** 2, axis=1)
    best = int(np.argmin(errors))
    return _ProfileFit(
        TailCurve("sigmoid", float(slopes[best]), float(ratios[best])),
        reference,
        float(levels[best]),
    )


def _select_growth_profile(values: np.ndarray) -> tuple[str, dict[str, float]]:
    """Train through 80–84 / 85–89; predict the next closed age group."""
    ages = np.array([72, 77, 82, 87, 92], dtype=float)
    scores = {}
    for kind in ("plateau", "exponent", "sigmoid"):
        errors = []
        for stop in (3, 4):
            fit = _fit_profile(ages[:stop], values[:stop], kind)
            error = float(fit.predict_log(ages[stop : stop + 1])[0] - np.log(values[stop]))
            errors.append(error)
        scores[kind] = float(np.sqrt(np.mean(np.square(errors))))
    # Prefer fewer parameters within 0.03 log units (~3% multiplicative error).
    best = min(scores.values())
    selected = next(kind for kind in scores if scores[kind] <= best + 0.03)
    return selected, scores


def _has_fit_support(table: pd.DataFrame) -> bool:
    """Require enough complete positive years without inventing zero-rate evidence."""
    complete = table.dropna()
    means = complete.mean()
    return bool(
        len(complete[(complete > 0).all(axis=1)]) >= 5
        and (means > 0).all()
        and means.iloc[-2:].mean() >= 5
        and (complete.gt(0).mean() >= 0.8).all()
    )


def _fit_dementia_tail(table: pd.DataFrame, period: str) -> CauseTailFit:
    """Use the agreed 70–89 exponent; retain the 70–94 sigmoid for comparison.

    Near-age predictive superiority of a sigmoid does not establish its remote
    plateau. The primary family is an explicit scenario choice, not a CV winner.
    Neither its slope nor its availability depends on the 90–94 observations.
    """
    younger = table[[70, 75, 80, 85]]
    if not _has_fit_support(younger):
        return CauseTailFit(
            TailCurve(),
            "—",
            period,
            "Fallback",
            "Insufficient positive pre-pandemic data at 70–89 for the dementia exponent.",
        )
    curve = _fit_profile(
        np.array([72, 77, 82, 87]), younger.dropna().mean().to_numpy(), "exponent"
    ).at_tail_anchor()
    alternative_curve = None
    alternative_score = None
    if _has_fit_support(table):
        complete = table.dropna()
        alternative_curve = _fit_profile(
            np.array([72, 77, 82, 87, 92]), complete.mean().to_numpy(), "sigmoid"
        ).at_tail_anchor()
        positive = complete[(complete > 0).all(axis=1)]
        alternative_score = float(
            np.mean([_select_growth_profile(row)[1]["sigmoid"] for row in positive.to_numpy()])
        )
    return CauseTailFit(
        curve,
        "70–89",
        period,
        "Scenario assumption",
        "Exponent fitted at 70–89 is the chosen dementia scenario. Near-age slowing does not "
        "establish a remote plateau; a sigmoid fitted at 70–94 is retained as a comparison when supported.",
        alternative="sigmoid" if alternative_curve else None,
        alternative_log_rmse=alternative_score,
        alternative_curve=alternative_curve,
        alternative_fit_ages="70–94" if alternative_curve else None,
    )


def fit_cause_tails(
    history: pd.DataFrame, codes: tuple[str, ...] | list[str]
) -> dict[str, CauseTailFit]:
    """Fit each requested cause without learning from 2020+ or the 95+ bucket.

    A reproducible late downturn is treated separately: two terminal closed
    groups identify a declining slope but cannot validate its continuation.
    Such selections are explicitly labeled as weakly validated.
    """
    history = history[history.age_start < 95].copy()
    history = merge_dementia(history) if "year" in history else history
    if "year" not in history:
        history["year"] = 2021
    source = history[history.year.between(2011, 2019) & history.age_start.between(70, 90)]
    fits = {}
    for code in codes:
        group = source[source.cause_code == code]
        years = sorted(group.year.unique())
        period = f"{years[0]}–{years[-1]}" if years else "unavailable"
        fallback_reason = None
        if code in COVID_CODES:
            fallback_reason = (
                "No pre-pandemic age profile; plateau age tail with a separate COVID time path."
            )
        elif not years:
            fallback_reason = "No pre-pandemic history supplied; plateau fallback."
        if fallback_reason:
            fits[code] = CauseTailFit(TailCurve(), "—", period, "Fallback", fallback_reason)
            continue
        table = group.pivot(index="year", columns="age_start", values="rate").reindex(
            columns=[70, 75, 80, 85, 90]
        )
        if code == DEMENTIA_CODE:
            fits[code] = _fit_dementia_tail(table, period)
            continue
        complete = table.dropna()
        positive = complete[(complete > 0).all(axis=1)]
        means = complete.mean().to_numpy(dtype=float)
        # Rates alone cannot distinguish a tiny population from strong evidence.
        # Avoid fitting logs after discarding zeros or adding arbitrary pseudocounts.
        if not _has_fit_support(table):
            fits[code] = CauseTailFit(
                TailCurve(),
                "—",
                period,
                "Fallback",
                "Sparse/zero rates or fewer than five complete years; counts and exposures are unavailable.",
            )
            continue
        decline_agreement = float((complete[90] < complete[85]).mean())
        if means[-1] < 0.97 * means[-2] and decline_agreement >= 0.75:
            slopes = -np.log(positive[90] / positive[85]) / 5
            slope = float(np.clip(slopes.median(), 0, MAX_SLOPE))
            fits[code] = CauseTailFit(
                TailCurve("decline", slope),
                "85–94",
                period,
                "Limited validation",
                "Repeated terminal decline; only two post-peak closed groups. Plateau remains a sensitivity alternative.",
                alternative="plateau",
                year_agreement=decline_agreement,
            )
            continue
        year_results = [_select_growth_profile(row) for row in positive.to_numpy()]
        # Equal weight per year; year-specific levels cancel in log prediction errors.
        scores = {
            kind: float(np.mean([result[1][kind] for result in year_results]))
            for kind in ("plateau", "exponent", "sigmoid")
        }
        best = min(scores.values())
        selected = next(kind for kind in scores if scores[kind] <= best + 0.03)
        agreement = float(np.mean([result[0] == selected for result in year_results]))
        # Require both a material error reduction and repeated year-wise support
        # before spending the extra sigmoid parameter.
        if selected == "sigmoid" and (
            scores["sigmoid"] > 0.8 * scores["exponent"] or agreement < 2 / 3
        ):
            selected = min(("plateau", "exponent"), key=scores.get)
            agreement = float(np.mean([result[0] == selected for result in year_results]))
        fit = _fit_profile(np.array([72, 77, 82, 87, 92]), means, selected)
        alternative = min((kind for kind in scores if kind != selected), key=scores.get)
        status = "Stable selection" if agreement >= 2 / 3 else "Ambiguous selection"
        reason = "Blocked-age validation across pre-pandemic years; simpler forms preferred on near ties."
        if selected == "sigmoid":
            status = "Plateau level uncertain"
            reason += " Saturation is bounded by a search grid, not measured at extreme ages."
        fits[code] = CauseTailFit(
            fit.at_tail_anchor(),
            "70–94",
            period,
            status,
            reason,
            scores[selected],
            alternative,
            scores[alternative],
            agreement,
        )
    return fits


def tail_fit_table(fits: dict[str, CauseTailFit], names: dict[str, str]) -> pd.DataFrame:
    rows = []
    for code, fit in fits.items():
        curve = fit.curve
        rows.append(
            {
                "Cause": names[code],
                "Code": code,
                "Family": curve.kind,
                "Age window": fit.fit_ages,
                "History": fit.history_years,
                "Slope (1/year)": curve.slope,
                "Sigmoid plateau / age-94 rate": curve.asymptote_ratio
                if curve.kind == "sigmoid"
                else None,
                "Validation log-RMSE": fit.validation_log_rmse,
                "Alternative": fit.alternative,
                "Alternative log-RMSE": fit.alternative_log_rmse,
                "Alternative age window": fit.alternative_fit_ages,
                "Alternative slope (1/year)": fit.alternative_curve.slope
                if fit.alternative_curve
                else None,
                "Alternative plateau / age-94 rate": fit.alternative_curve.asymptote_ratio
                if fit.alternative_curve and fit.alternative_curve.kind == "sigmoid"
                else None,
                "Year agreement": fit.year_agreement,
                "Status": fit.status,
                "Reason": fit.reason,
            }
        )
    return pd.DataFrame(rows).sort_values(["Family", "Cause"]).reset_index(drop=True)
