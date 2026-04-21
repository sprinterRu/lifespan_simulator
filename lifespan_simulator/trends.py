"""Historical trend utilities for the explorer tab."""

from __future__ import annotations

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


def build_trend_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit a linear trend with a 95% confidence interval for the mean rate."""
    ordered = frame.sort_values("year").copy()
    x = ordered["year"].to_numpy(dtype=float)
    y = ordered["rate"].to_numpy(dtype=float)
    n_obs = len(ordered)

    if n_obs < 3:
        ordered["trend"] = y
        ordered["trend_lower"] = y
        ordered["trend_upper"] = y
        return ordered, {"slope": float("nan"), "r_squared": float("nan")}

    design = np.column_stack([np.ones_like(x), x])
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefficients
    residuals = y - fitted
    dof = n_obs - 2
    rss = float(np.sum(residuals**2))
    t_critical = T_CRITICAL_95.get(dof, 1.96)

    if dof <= 0 or np.isclose(rss, 0.0):
        lower = fitted
        upper = fitted
    else:
        sigma = np.sqrt(rss / dof)
        x_mean = float(np.mean(x))
        sxx = float(np.sum((x - x_mean) ** 2))
        se_mean = sigma * np.sqrt((1 / n_obs) + ((x - x_mean) ** 2) / sxx)
        margin = t_critical * se_mean
        lower = np.clip(fitted - margin, a_min=0.0, a_max=None)
        upper = fitted + margin

    total_sum_squares = float(np.sum((y - np.mean(y)) ** 2))
    if np.isclose(total_sum_squares, 0.0):
        r_squared = 1.0
    else:
        r_squared = 1 - (rss / total_sum_squares)

    ordered["trend"] = fitted
    ordered["trend_lower"] = lower
    ordered["trend_upper"] = upper

    return ordered, {"slope": float(coefficients[1]), "intercept": float(coefficients[0]), "r_squared": r_squared}

