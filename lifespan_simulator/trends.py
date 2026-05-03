"""Historical trend utilities for the explorer tab."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.16,
    14: 2.145,
    15: 2.131,
    16: 2.12,
    17: 2.11,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.08,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.06,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


@dataclass(frozen=True)
class TrendModel:
    """Significant linear mortality trends used by the simulator.

    Attributes:
        slope_table: Nested lookup keyed first by ``age_start`` and then by
            ``cause_code``. Values are annual mortality-rate slopes per 100,000
            population. Non-significant trends are stored as ``0.0`` so the
            simulator can use one simple lookup for every cause-age pair.
        significant_count: Number of cause-age regressions whose 95% slope
            confidence interval excludes zero.
        total_count: Number of cause-age regressions fitted for the simulator's
            non-overlapping cause set.
    """

    slope_table: dict[int, dict[str, float]]
    significant_count: int
    total_count: int


def build_trend_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit a linear mortality-rate trend and regression summary statistics.

    Args:
        frame: Mortality DataFrame for one selected cause and one selected age
            group. It is sliced in ``app.render_explorer_tab`` from the complete
            normalized mortality dataset and must include ``year`` and ``rate``
            columns.

    Returns:
        A pair containing the ordered trend DataFrame and a statistics mapping.
        The DataFrame preserves the input rows sorted by year and adds
        ``trend``, ``trend_lower``, and ``trend_upper`` columns for the fitted
        mean line and its 95% confidence interval. The stats mapping contains
        ``slope``, ``intercept``, ``r_squared``, ``slope_ci_lower``,
        ``slope_ci_upper``, and ``slope_significant``. ``slope_significant`` is
        true when the 95% confidence interval for the slope excludes zero.
    """
    ordered = frame.sort_values("year").copy()
    x = ordered["year"].to_numpy(dtype=float)
    y = ordered["rate"].to_numpy(dtype=float)
    n_obs = len(ordered)

    if n_obs < 3:
        ordered["trend"] = y
        ordered["trend_lower"] = y
        ordered["trend_upper"] = y
        return ordered, {
            "slope": float("nan"),
            "intercept": float("nan"),
            "r_squared": float("nan"),
            "slope_ci_lower": float("nan"),
            "slope_ci_upper": float("nan"),
            "slope_significant": False,
        }

    design = np.column_stack([np.ones_like(x), x])
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefficients
    residuals = y - fitted
    dof = n_obs - 2
    rss = float(np.sum(residuals**2))
    t_critical = T_CRITICAL_95.get(dof, 1.96)
    x_mean = float(np.mean(x))
    sxx = float(np.sum((x - x_mean) ** 2))
    slope = float(coefficients[1])

    if dof <= 0 or np.isclose(rss, 0.0) or np.isclose(sxx, 0.0):
        lower = fitted
        upper = fitted
        slope_ci_lower = slope
        slope_ci_upper = slope
    else:
        sigma = np.sqrt(rss / dof)
        se_mean = sigma * np.sqrt((1 / n_obs) + ((x - x_mean) ** 2) / sxx)
        margin = t_critical * se_mean
        lower = np.clip(fitted - margin, a_min=0.0, a_max=None)
        upper = fitted + margin
        # In simple linear regression, Var(slope) = MSE / Sxx. The same
        # t-critical value used for the mean confidence band gives the 95% CI.
        slope_margin = t_critical * np.sqrt((rss / dof) / sxx)
        slope_ci_lower = slope - slope_margin
        slope_ci_upper = slope + slope_margin

    total_sum_squares = float(np.sum((y - np.mean(y)) ** 2))
    if np.isclose(total_sum_squares, 0.0):
        r_squared = 1.0
    else:
        r_squared = 1 - (rss / total_sum_squares)

    ordered["trend"] = fitted
    ordered["trend_lower"] = lower
    ordered["trend_upper"] = upper

    return ordered, {
        "slope": slope,
        "intercept": float(coefficients[0]),
        "r_squared": r_squared,
        "slope_ci_lower": float(slope_ci_lower),
        "slope_ci_upper": float(slope_ci_upper),
        "slope_significant": bool(slope_ci_lower > 0 or slope_ci_upper < 0),
    }


def build_trend_model(raw_frame: pd.DataFrame, simulator_frame: pd.DataFrame) -> TrendModel:
    """Build significant cause-age trend slopes for survival simulation.

    Args:
        raw_frame: Complete normalized mortality dataset returned by
            ``load_mortality_data``. It contains all available years, including
            the 2011-2021 history used to fit each linear trend.
        simulator_frame: 2021 non-overlapping simulator DataFrame returned by
            ``get_simulator_frame``. It defines the exact age buckets and cause
            codes that can contribute to survival totals without double-counting
            aggregate categories.

    Returns:
        :class:`TrendModel` containing one slope entry for every simulator
        age-bucket and cause-code pair. A slope is retained only when the
        corresponding 95% confidence interval excludes zero; otherwise the
        stored slope is ``0.0``.
    """
    simulator_keys = simulator_frame[["age_start", "cause_code"]].drop_duplicates()
    trend_source = raw_frame.merge(simulator_keys, on=["age_start", "cause_code"], how="inner")

    slope_table: dict[int, dict[str, float]] = {}
    significant_count = 0
    total_count = 0

    for (age_start, cause_code), group in trend_source.groupby(["age_start", "cause_code"], sort=True):
        _, stats = build_trend_frame(group)
        total_count += 1

        slope = 0.0
        if bool(stats["slope_significant"]) and not pd.isna(stats["slope"]):
            slope = float(stats["slope"])
            significant_count += 1

        slope_table.setdefault(int(age_start), {})[str(cause_code)] = slope

    return TrendModel(
        slope_table=slope_table,
        significant_count=significant_count,
        total_count=total_count,
    )
