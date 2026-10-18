# Validation snapshot

Reference population: males in Spain, 2021. See `data/reference_metadata.json` for exact requests, retrieval time, source revisions, units and hashes. The checked-in inputs allow offline reproduction.

## Source cause totals

After excluding overlapping categories, the local source contains 69 causes across 13 age groups. The six directly checkable roll-up identities agree within 0.02 deaths per 100,000 (rounding). The official all-cause audit is performed before any calibration.

The largest absolute discrepancy is **−3.51 per 100,000**, at 65–69. The largest relative discrepancy is **−0.587%**, at 50–54. At 95+, the source is **+0.03 per 100,000** above the official total. The mismatch is displayed rather than attributed to a specific cause without evidence. Original extraction metadata for `mortality.csv` are unavailable.

## Period life expectancy

To compare with a period life table, disable trends, retain the 2021 COVID level, and do not apply interventions. Calibration uses published Mx; the ex comparison checks reconstruction and old-age continuation. It is not an independent test of future forecasting skill.

| Starting age | Official remaining years | Source + flat 95+ | Source + Kannisto | Calibrated + Kannisto | Calibrated error |
|---|---:|---:|---:|---:|---:|
| 35 | 45.9 | 45.809 | 45.773 | 45.896 | −0.004 |
| 65 | 18.9 | 18.891 | 18.850 | 18.897 | −0.003 |
| 85 | 6.1 | 6.249 | 6.163 | 6.086 | −0.014 |
| 95 | 3.0 | 3.258 | 2.808 | 2.763 | −0.237 |

The official ex values are rounded to 0.1 years. The age-95 discrepancy remains material: the open 95+ rate cannot identify the true single-age curve, and the chosen continuation differs from the reference table's closure. Tests allow 0.1-year differences below 95 and 0.3 years at 95 for this snapshot. Those are regression tolerances, not claims of scientific accuracy.

## Illustrative scenario outputs

For a cohort aged 35 in 2021, the current defaults (calibrated Kannisto, 2011–2019 trends, 50-year trend horizon, fading COVID) give mean age **84.536**, median **87.386**, and p99 **102.555** without interventions.

Applying the standard two-stage intervention to every cause gives:

- Median age **90.550**; p90 **106.043**; p95 **118.788**.
- **56.659 expected remaining years lived before age 200** (restricted mean).
- **2.198% alive at age 200**.
- An upper bound of **2.507 additional expected life-years** after that horizon.
- Full mean and p99 marked **unresolved / beyond horizon**.

These are conditional model outputs, not predictions of medical progress. The changes deliberately alter the previous defaults (pandemic treatment, trend fitting window/horizon, calibration and older-age continuation), so the old/new output difference should not be attributed to a single assumption.

## Automated checks

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

The suite covers analytical constant-hazard integrals and percentiles, monotone survival, conservation of deaths plus survivors, no-op intervention equivalence, agreement between charts and survival, intervention boundary years, stable tiny/100% reductions, finite computation, tail bounds and unresolved extreme scenarios, reference checksums and calibration, older-age continuity, explicit COVID decay, leakage-free pre-pandemic trend preparation, CSV round trips and non-finite input rejection, and Streamlit interactions with scenarios and sensitivity results.

GitHub Actions repeats the checks on Python 3.10 and 3.13. No test refreshes data or contacts Eurostat.

## Remaining work

- Out-of-sample forecast evaluation over multiple historical origins, plus improved statistical trend models.
- Observed death counts and exposure weights for model fitting and sampling uncertainty.
- More detailed late-life data and an explicit comparison of alternative tail closures.
- Age-dependent treatment uptake/effectiveness, disease states and shared individual risk factors.
- A probabilistic uncertainty model; the present sensitivity analysis is deterministic.
