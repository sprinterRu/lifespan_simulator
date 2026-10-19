"""Reviewed cause-specific age tails with explicit parameter provenance.

Nearby slopes use pre-pandemic rates. Families, slowdown timescales and residuals
are scenario assumptions, not statistically validated extreme-age predictions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .cause_models import CAUSE_ASSUMPTIONS
from .config import COVID_CODES, DECLINE_RESIDUAL_DEFAULT
from .data import DEMENTIA_CODE, merge_dementia

TAIL_ANCHOR_AGE = 94
LAST_CLOSED_MIDPOINT = 92
TailKind = Literal["power", "bounded", "plateau", "decline", "linked", "exponent", "sigmoid"]
SLOWDOWN_YEARS_DEFAULT = 20.0
FAMILY_LABELS = {
    "power": "Slowing growth (P)",
    "bounded": "Bounded growth (B)",
    "plateau": "Flat (F)",
    "decline": "Decline with residual (D)",
    "linked": "Linked reporting category (L)",
    "exponent": "Exponential (comparison)",
    "sigmoid": "Sigmoid (comparison)",
}
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
    slowdown_years: float = SLOWDOWN_YEARS_DEFAULT
    linked_group: Literal["natural", "external"] | None = None

    def __post_init__(self) -> None:
        if self.kind not in FAMILY_LABELS:
            raise ValueError("Unknown cause-tail family")
        if not np.isfinite(self.slope) or self.slope < 0:
            raise ValueError("Tail slope must be finite and non-negative")
        if not np.isfinite(self.asymptote_ratio) or self.asymptote_ratio < 1:
            raise ValueError("Increasing sigmoid asymptote must be at least its anchor")
        if not np.isfinite(self.slowdown_years) or self.slowdown_years <= 0:
            raise ValueError("Slowdown timescale must be finite and positive")
        if self.kind == "linked" and self.linked_group not in {"natural", "external"}:
            raise ValueError("Linked curves require a natural or external reference group")

    def log_multiplier(
        self, elapsed: np.ndarray, slope_multiplier: float = 1, decline_residual: float = 0
    ) -> np.ndarray:
        """Normalized to exactly one at the splice; safe at remote ages."""
        distance = np.asarray(elapsed, dtype=float)
        if self.kind == "linked":
            raise ValueError("Linked trajectories must be evaluated with their underlying causes")
        growth = self.slope * slope_multiplier * distance
        if self.kind == "power":
            return (
                self.slope
                * slope_multiplier
                * self.slowdown_years
                * np.log1p(distance / self.slowdown_years)
            )
        if self.kind == "bounded":
            return (
                -self.slope
                * slope_multiplier
                * self.slowdown_years
                * np.expm1(-distance / self.slowdown_years)
            )
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
    description: str = ""
    rationale: str = ""
    provisional: bool = False

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
    """Reconstruct the legacy dementia fits for the descriptive comparison.

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


def _fit_descriptive_tails(
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


def fit_cause_tails(
    history: pd.DataFrame, codes: tuple[str, ...] | list[str]
) -> dict[str, CauseTailFit]:
    """Apply the reviewed families; estimate only their nearby slopes from data.

    The 20-year slowdown and 25% declining residual are scenario assumptions.
    Sparse positive slopes get 25% own-profile / 75% group-median weight.
    COVID borrows the infection-group median, identically for all three codes.
    Group pools use all available registered causes, independent of requested codes.
    """
    history = history[history.age_start < 95].copy()
    history = merge_dementia(history) if "year" in history else history
    if "year" not in history:
        history["year"] = 2021
    source = history[history.year.between(2011, 2019) & history.age_start.between(70, 90)]
    tables = {
        code: group.pivot(index="year", columns="age_start", values="rate").reindex(
            columns=[70, 75, 80, 85, 90]
        )
        for code, group in source.groupby("cause_code")
    }
    slopes, supported = {}, {}
    for code, table in tables.items():
        supported[code] = _has_fit_support(table)
        means = table[[80, 85, 90]].dropna().mean().to_numpy(dtype=float)
        if np.isfinite(means).all() and (means > 0).all():
            slopes[code] = float(
                np.clip(np.polyfit([82, 87, 92], np.log(means), 1)[0], 0, MAX_SLOPE)
            )
    pools = {}
    for code, assumption in CAUSE_ASSUMPTIONS.items():
        if supported.get(code) and assumption.family in {"power", "bounded"}:
            pools.setdefault(assumption.group, []).append(slopes[code])
    group_slopes = {group: float(np.median(values)) for group, values in pools.items()}

    # Keep the original descriptive selector available for synthetic/custom data.
    unknown = [code for code in codes if code not in CAUSE_ASSUMPTIONS]
    fits = _fit_descriptive_tails(history, unknown) if unknown else {}
    for code in codes:
        if code not in CAUSE_ASSUMPTIONS:
            continue
        assumption = CAUSE_ASSUMPTIONS[code]
        table = tables.get(code)
        years = sorted(table.index) if table is not None else []
        period = f"{years[0]}–{years[-1]}" if years else "unavailable"
        kind = assumption.family
        slope, window = 0.0, "—"
        basis = "No additional age slope; the observed baseline level is retained."
        if kind in {"power", "bounded"}:
            if supported.get(code):
                slope, window = slopes[code], "80–94"
                basis = "Initial log-slope estimated from mean 2011–2019 rates at ages 80–94."
            else:
                group_slope = group_slopes.get(assumption.group, 0.05)
                slope = (
                    0.25 * slopes[code] + 0.75 * group_slope
                    if code in slopes and code not in COVID_CODES
                    else group_slope
                )
                window = "80–94 (pooled)" if assumption.group in group_slopes else "assumed"
                basis = (
                    "Sparse slope: 25% own mean-profile slope and 75% group median."
                    if code in slopes and code not in COVID_CODES
                    else "No supported own slope; borrowed the group's supported median slope."
                    if assumption.group in group_slopes
                    else "No supported group slope; assumed 0.05/year."
                )
            basis += " Slowdown timescale: 20 years, assumed rather than fitted."
        elif kind == "decline":
            if table is not None:
                terminal = table[[85, 90]].dropna().mean().to_numpy(dtype=float)
                if np.isfinite(terminal).all() and (terminal > 0).all():
                    local_decline = float(
                        np.clip(-np.log(terminal[1] / terminal[0]) / 5, 0, MAX_SLOPE)
                    )
                    slope = local_decline / (1 - DECLINE_RESIDUAL_DEFAULT)
                    window = "85–94"
            basis = (
                "Decay parameter matches the mean 85–94 log-decline at a 25% residual. "
                "The residual is an editable assumption, not a fitted floor."
                if window != "—"
                else "No supported decline slope; hold the level until data are supplied."
            )
        elif kind == "linked":
            basis = (
                f"Follows identified {assumption.group} causes, including their calendar and treatment effects. "
                "Classification ratios use 2021 age-specific rates and are held at the age-94 boundary thereafter."
            )
        curve = TailCurve(kind, slope, linked_group=assumption.group if kind == "linked" else None)
        alternative = None
        # Preserve the formerly available dementia sigmoid as an explicitly
        # descriptive comparison, without giving the new prior its CV score.
        if code == DEMENTIA_CODE and table is not None and _has_fit_support(table):
            alternative = _fit_dementia_tail(table, period)
        fits[code] = CauseTailFit(
            curve=curve,
            fit_ages=window,
            history_years=period,
            status="Provisional assumption" if assumption.provisional else "Scenario assumption",
            reason=basis,
            alternative="sigmoid" if alternative and alternative.alternative_curve else None,
            alternative_curve=alternative.alternative_curve if alternative else None,
            alternative_fit_ages=alternative.alternative_fit_ages if alternative else None,
            alternative_log_rmse=alternative.alternative_log_rmse if alternative else None,
            description=assumption.description,
            rationale=assumption.rationale,
            provisional=assumption.provisional,
        )
    return {code: fits[code] for code in codes}


def tail_fit_table(
    fits: dict[str, CauseTailFit],
    names: dict[str, str],
    decline_residual: float = DECLINE_RESIDUAL_DEFAULT,
) -> pd.DataFrame:
    rows = []
    for code, fit in fits.items():
        curve = fit.curve
        rows.append(
            {
                "Cause": names[code],
                "Code": code,
                "Family": curve.kind,
                "Model": FAMILY_LABELS[curve.kind],
                "Description": fit.description,
                "Why this model": fit.rationale,
                "Provisional": fit.provisional,
                "Age window": fit.fit_ages,
                "History": fit.history_years,
                "Slope (1/year)": curve.slope,
                "Slowdown (years, assumed)": curve.slowdown_years
                if curve.kind in {"power", "bounded"}
                else None,
                "Linked group": curve.linked_group,
                "Declining residual (assumed)": decline_residual
                if curve.kind == "decline"
                else None,
                "Bounded plateau / age-94 rate (assumed)": float(
                    np.exp(curve.slope * curve.slowdown_years)
                )
                if curve.kind == "bounded"
                else None,
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
