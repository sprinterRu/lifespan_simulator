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

With the floor applying only to slow improvements, applying the standard intervention to every cause gives an analytically **infinite mean**. The median is **90.550**, p90 **106.043**, and p95 **118.788**. At least **3.687%** of the cohort retains positive limiting survival under the indefinite fast-reduction assumption, so p99 is infinite. Moving fast to 2065 for every cause gives median **98.285** and limiting survival at least **31.683%**; p75 and later percentiles are infinite. These statements concern the mathematical model, not real-world immortality.

For finite life-extension examples, apply the same settings to every cause **except falls (`W00-W19`)**, retaining that residual risk:

| Fast start | Mean at age-200 horizon | Mean at age-2,000 horizon | Median | p99 | Remaining mean bound at 2,000 |
|---|---|---:|---:|---:|---:|
| 2080 | Unresolved | 92.928 | 90.461 | 262.094 | 6.7 × 10⁻⁷ years |
| 2065 | Unresolved | 129.936 | 97.219 | 518.332 | 5.6 × 10⁻⁶ years |

Increasing the horizon again to 10,000 changes either finite mean by less than 0.00001 years.

For dementia alone, slow 1.5% from 2040, fast 99.9% from 2065, and slow floor 1% give **0.1193 deaths per 100,000** at age 79, **0.0001645** at age 80, and approximately **7.30 × 10⁻⁹³** at age 110. The former 1% cap no longer stops fast reductions.

These are conditional model outputs, not predictions of medical progress. The changes deliberately alter the previous defaults (pandemic treatment, trend fitting window/horizon, calibration and older-age continuation), so the old/new output difference should not be attributed to a single assumption.

## Automated checks

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

The suite covers analytical constant-hazard integrals and percentiles, including a 500-year expected remaining lifetime integrated to age 10,000; monotone survival; conservation of deaths plus survivors; no-op intervention equivalence; agreement between charts and survival; intervention boundary years; stable tiny/100% reductions; Gompertz overflow and excluded causes at remote ages; tail bounds and unresolved extreme scenarios; reference checksums and calibration; older-age continuity; explicit COVID decay; leakage-free pre-pandemic trend preparation; CSV round trips and non-finite input rejection; and Streamlit interactions with scenarios, horizons and sensitivity results.

GitHub Actions repeats the checks on Python 3.10 and 3.13. No test refreshes data or contacts Eurostat.

## Remaining work

- Out-of-sample forecast evaluation over multiple historical origins, plus improved statistical trend models.
- Observed death counts and exposure weights for model fitting and sampling uncertainty.
- More detailed late-life data and an explicit comparison of alternative tail closures.
- Age-dependent treatment uptake/effectiveness, disease states and shared individual risk factors.
- A probabilistic uncertainty model; the present sensitivity analysis is deterministic.
