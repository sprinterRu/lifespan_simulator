# Validation snapshot

Reference population: males in Spain, 2021. See `data/reference_metadata.json` for exact requests, retrieval time, source revisions, units and hashes. The checked-in inputs allow offline reproduction. All 95+ observations are excluded from the calculations below.

## Source cause totals

After excluding overlapping categories and merging dementia with Alzheimer disease, the simulator contains 68 causes across 12 closed age groups (35–39 through 90–94). The merge preserves the total in every age/year cell. The official all-cause audit is performed before calibration.

The largest absolute discrepancy is **−3.51 per 100,000**, at 65–69. The largest relative discrepancy is **−0.587%**, at 50–54. The mismatch is displayed rather than attributed to a specific cause without evidence. Original extraction metadata for `mortality.csv` are unavailable.

## Period life expectancy

To compare with a period life table, disable trends, retain the 2021 COVID level, and do not apply interventions. Calibration uses published Mx; the ex comparison checks reconstruction and old-age continuation. It is not an independent test of future forecasting skill. Individual curves use the default dementia exponent, automatic fits for other causes, and a zero declining-tail residual.

| Starting age | Official remaining years | Source + flat after 94 | Source + Kannisto | Calibrated + Kannisto | Calibrated + individual | Individual error |
|---|---:|---:|---:|---:|---:|---:|
| 35 | 45.9 | 46.004 | 45.878 | 45.896 | 45.884 | −0.016 |
| 65 | 18.9 | 19.110 | 18.969 | 18.897 | 18.883 | −0.017 |
| 85 | 6.1 | 6.703 | 6.410 | 6.086 | 6.058 | −0.042 |
| 94 | 3.1 | 4.987 | 3.848 | 2.975 | 2.856 | −0.244 |

Official ex values are rounded to 0.1 years. The age-94 discrepancies remain visible because extrapolated mortality after 94 differs from the reference table's closure. Numerical tests use regression tolerances of 0.1 years at 35/65/85 and 0.3 at 94, not claims of scientific accuracy. No open-age Mx or ex is used to tune the tail.

## Individual shape selections

The default configuration has **38 exponents, 9 sigmoids, 18 plateaus and 3 declines**. Dementia's exponent is an explicit scenario choice; other causes follow the selector. Of the 18 plateaus, 14 are explicit fallbacks for sparse/zero rates or absent pre-pandemic history. Automatic scores use two blocked-age predictions within each pre-pandemic year. They do not validate ages beyond the last closed group.

Examples from the default configuration:

| Cause | Family | Slope / year | Interpretation |
|---|---|---:|---|
| Lung cancer | Decline | 0.03853 | Terminal decrease in all nine years; only two post-peak groups, no independent tail validation |
| Liver cancer | Decline | 0.01382 | Terminal decrease in seven of nine years; limited evidence |
| Chronic liver disease | Decline | 0.05904 | Terminal decrease in all nine years; limited evidence |
| Prostate cancer | Exponent | 0.12072 | Simpler curve within the selection tolerance of a sigmoid |
| Bladder cancer | Exponent | 0.09320 | Ambiguous selection; inspect the sigmoid alternative |
| Dementia including Alzheimer disease | Exponent | 0.19801 | Chosen 70–89 extrapolation; doubling time 3.50 years |
| HIV | Plateau fallback | — | Too sparse for a supported positive-rate fit |

Thus cancers do not share a forced common shape. Dementia's default exponent has no automatic selection score. The former 70–94 sigmoid (slope 0.21674, asymptote ratio 1.5287) is preserved as a selectable comparison and plotted alongside it. That curve's log-RMSE of 0.11652 concerns next-group predictions below 95; it does not establish a biological ceiling. Neither curve models incidence.

## Illustrative scenario outputs

A cohort aged 35 in 2021, with default individual tails including the dementia exponent, calibrated levels, 2011–2019 calendar trends, a 50-year trend horizon and fading COVID. The following table retains a fixed **raw 10% fast reduction**, for comparison with the previous UI defaults:

| Intervention on every cause | Mean age at death | Median | p99 |
|---|---:|---:|---:|
| None | 84.578 | 87.650 | 101.343 |
| Slow 1.5% from 2040, fast 10% from 2080 | 88.272 | 90.744 | 113.216 |
| Slow 1.5% from 2040, fast 10% from 2065 | 96.461 | 98.113 | 132.144 |

These interventions retain the 1% slow-stage floor and have no fast-stage floor. Several exponent causes grow faster than fast treatment suppresses them: the threshold is `-log(0.9) ≈ 0.10536/year`. Consequently, these individual-tail scenarios still have finite mean lifespans. Removing the fast floor does not by itself imply indefinite survival.

The new UI suggestions use `1 - 0.9*exp(-b)` for each active exponent, retaining 10% for other families. With slow start 2040 and fast start 2080, selecting the **38 exponent causes only** gives mean age **89.856**, median **89.884** and p99 **160.217**. Applying the new suggestions to **all causes** gives an analytically infinite mean, median **90.744** and limiting survival at least **12.156%**. Every exponential term is now outpaced by treatment; the infinite mean still follows from indefinite progress under the selected model, not observed human survival.

The current UI defaults to fast start **2070**. With that date and the age-compensated suggestions, selecting the 38 exponent causes gives mean age **98.123**, median **93.538** and p99 **192.546**. Selecting all causes gives an analytically infinite mean, median **114.610** and limiting survival at least **49.061%**. The 2080 examples above retain the earlier date to separate the effect of rate compensation from the effect of starting treatment sooner.

For comparison, the common Kannisto mode with the same calendar settings gives a baseline mean of **84.627**. Applying slow 1.5% from 2040 and raw fast 10% from 2080 to every cause yields an analytically infinite mean, with median **90.744** and limiting survival at least **3.552%**. This difference demonstrates dependence on remote-age assumptions, not evidence favoring infinite or finite human lifespans.

The defaults changed from a shared tail to individual tails, dementia was merged before calendar trend fitting, and 95+ inputs were removed. Changes relative to older snapshots cannot be attributed to a single assumption. All outputs are conditional model results, not predictions of medical progress.

## Automated checks

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

The suite covers analytical constant-hazard integrals and percentiles, including a 500-year expected remaining lifetime integrated to age 10,000; monotone survival; conservation of deaths plus survivors; no-op intervention equivalence; agreement between charts and survival; intervention boundary years; tiny/100% reductions; remote-age overflow protection; numerical tail bounds and unresolved scenarios; reference checksums/calibration; explicit COVID decay; and CSV round trips and invalid-input rejection.

New checks cover dementia aggregation and safe legacy CSV migration, exclusion of 95+ even when arbitrary open-group records are supplied, pre-pandemic-only curve selection, known synthetic shape recovery, continuity of all four curve families, independent cause hazards after elimination, analytical declining-tail bounds with and without residuals, COVID/residual splitting, fast treatment versus exponent growth, and Streamlit trajectory overrides and controls.

Age-compensated defaults are checked against the full mortality surface: consecutive exponential hazards must differ by exactly 10% under fixed calendar factors, in both individual and common Gompertz modes. UI tests cover bulk selection, default recalculation when changing dementia's shape, preservation of manual/imported reductions and explicit reset.

GitHub Actions repeats the checks on Python 3.10 and 3.13. No test refreshes data or contacts Eurostat.

## Remaining work

- Death counts and population exposures for weighted fits and sampling uncertainty.
- Independent validation of late-age cause-specific shapes, including cohort/coding effects and terminal declines.
- Robustness to fit-window and curve-search bounds, beyond the current deterministic sensitivity comparisons.
- Out-of-sample calendar forecasting, hierarchical trends and correction for multiple testing.
- Age-dependent treatment uptake/effectiveness, disease states and correlated individual risks.
- A probabilistic uncertainty model; current sensitivity ranges are not confidence intervals.
