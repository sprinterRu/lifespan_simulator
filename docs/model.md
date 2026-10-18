# Model and assumptions

## Mortality surface

Let `r_c(a)` denote the interpolated 2021 rate for cause `c`, per 100,000. Closed five-year observations are interpolated across their age midpoints. All 95+ observations are excluded from every projection mode, calibration, trend fit and reference comparison. Raw snapshots remain unchanged. The last closed group's midpoint is 92 (90–94); source shares and time-trend slopes are held at that group beyond 92. Official all-cause calibration still varies through age 94. Independent age multipliers apply after 94.

The disjoint underlying-cause codes `F01_F03` and `G30` are summed into `F01_F03_G30` before fitting either age shapes or calendar trends. The simulator has 68 causes. This sums recorded causes of death; it does not infer incidence, prevalence or a clinical dementia diagnosis. Missing component cells are rejected rather than filled with zeros. The historical explorer preserves the two original series, excluding 95+.

For calibrated projections, the total baseline rate is `100000 × Mx(a)` for ages 35–94, from the official Eurostat life table. Every cause receives the same age-specific factor:

```text
calibration_factor(a) = official_total_rate(a) / sum_c r_c(a)
```

This preserves the source cause shares. It is an explicit modeling choice, not evidence that every cause-specific rate is correct. The source audit compares unmodified five-year rates against the *separate* Eurostat cause-of-death all-cause series. Life-table and cause-of-death production methods differ, so those two reference series are not forced to agree.

Uncalibrated mode uses the source totals instead. It is included for comparison and sensitivity analysis.

## Older ages

The default model selects an individual trajectory for every cause. For `t = age - 94 >= 0`, each age-94 cause rate is multiplied by one of four simple shapes:

| Family | Multiplier | Free shape parameters |
|---|---|---|
| Exponent | `exp(b*t)` | Nonnegative slope `b` |
| Sigmoid | `R / (1 + (R - 1)*exp(-b*t))` | Slope `b`, asymptote ratio `R > 1` |
| Plateau | `1` | None |
| Decline | `exp(-b*t)` | Nonnegative slope `b` |

Every multiplier equals one at the splice. Causes evolve independently after 94; their sum is **not** rescaled to an external all-cause curve. Cause shares can therefore change with age. Calendar trends, COVID and interventions multiply each cause's own age curve. An optional declining-tail residual `r` replaces its multiplier with `r + (1-r)*exp(-b*t)`; the default is zero. This is a baseline age-shape assumption, distinct from the slow-intervention floor. Fast interventions can reduce even the residual to zero.

### First-version shape selection

Only pre-pandemic rates from 2011–2019 and closed groups 70–74 through 90–94 are used. The midpoint representation approximates group-average rates; it is not a single-age observation or exposure-weighted fit. The baseline level is separately anchored to 2021. No data from 95+ or 2020–2021 influence shape selection.

1. Fewer than five complete, strictly positive years, an average rate below 5 per 100,000 in the last two groups, or fewer than 80% positive values in any group trigger an explicit plateau fallback. COVID also uses a fallback because it has no pre-pandemic series. These are pragmatic safeguards, not sampling-uncertainty estimates.
2. A last-group mean below 97% of the preceding group, with a decline in at least 75% of complete years, selects a terminal declining branch. Its slope is the median yearly negative log-ratio divided by five, clipped to [0, 0.4]. Only the two terminal groups identify it. It has **no independent post-peak validation score**, and its continuation is labeled limited evidence. A residual share or a manual plateau provides a sensitivity alternative.
3. Otherwise, compare plateau, exponent and sigmoid separately in each positive year: train on 70–84 and predict 85–89, then train on 70–89 and predict 90–94. Score each family by the mean across years of the two-point log-RMSE. The level is fitted separately in log space. Prefer the simpler family within 0.03 log units of the best score.
4. A sigmoid must additionally reduce the exponent's score by at least 20% and be selected in at least two thirds of years. Otherwise choose plateau or exponent. Exponent slopes are clipped to [0, 0.4]; sigmoid grid search uses 45 slopes in [0.02, 0.4] and 55 asymptote ratios in [1.02, 50], logarithmically spaced. Grid ratios are relative to the last training midpoint, then transformed to the splice at 94.
5. Refit the selected shape to the mean pre-pandemic profile. Record its nearest alternative, prediction scores, yearly agreement and limitations. A sigmoid's remote saturation level is uncertain even if its near-age predictions improve. Manual overrides retain the automatic alternative for comparison and clear the selected model's validation score.

This is an auditable descriptive selector, not an epidemiological causal model. Adjacent years share cohorts and cannot be treated as independent replications. Selection scores do not give confidence intervals, and evidence below 95 cannot validate centuries-long extrapolation. Sparse series and rates without death counts/exposures limit what can be inferred. No onset or disease-duration model is introduced.

### Dementia scenario choice

For combined dementia, the primary tail explicitly uses an exponent fitted to the mean 2011–2019 profile in the four closed groups **70–74 through 85–89**. Its log-slope is 0.198014/year, corresponding to a doubling time of 3.50 years. The 90–94 group neither affects this slope nor determines whether enough data are available to fit it. The level at the splice remains anchored to the baseline 2021 surface, as for other causes.

The previous sigmoid fitted at 70–94 is retained with its full parameters (slope 0.216741/year and age-94 asymptote ratio 1.528725), fit window and blocked-age score. It is shown as a comparison curve and can be selected as the active dementia scenario. Both choices are exported. The primary exponent is labeled **Scenario assumption**, with no automatic selection score or year-agreement claim. Better near-age prediction by the sigmoid does not validate its remote plateau. Other causes retain the selection procedure above.

### Common-curve comparators

The legacy Kannisto mode fits a nonnegative slope to `log(Mx / (1 - Mx))` at ages 80–94 and anchors the extrapolation at the observed age-94 rate:

```text
mu(a) = 1 / (1 + (1 / mu(94) - 1) × exp(-b × (a - 94)))
```

The asymptote is 1 per year. The fit is unweighted least squares, **not** the exposure-weighted likelihood estimation used in the [Human Mortality Database methods protocol](https://www.mortality.org/File/GetDocument/Public/Docs/MethodsProtocolV6.pdf). We do not have the requisite observed deaths and exposures in the source CSV.

Alternative models are:

- Gompertz: fit a non-negative slope to `log(Mx)` over 80–94, then `mu(a) = mu(94) × exp(b × (a-94))`.
- Flat after 94: keep the age-94 total constant, as a diagnostic comparator. The open 95+ bucket is not used.

The same procedures use source totals when calibration is disabled. In these three comparison modes, projected totals are allocated to the last closed group's fixed cause shares before trend, COVID and intervention adjustments. Interventions can lower the old-age asymptote; this does not impose a biological lifespan limit.

## Secular trends and pandemic separation

The default secular fit uses 2011–2019, excluding all three COVID categories. It retains linear slopes whose individual 95% intervals exclude zero. Multiple-testing correction, temporal dependence, hierarchical shrinkage and predictive calibration remain future work. The historical explorer still fits whichever complete series it displays, including 2020–2021; that view is descriptive.

For each cause, the source rate is advanced by `slope × elapsed`, where:

```text
elapsed = min(max(calendar_year - 2021, 0), trend_horizon_years)
```

The projected source rate is bounded relative to its age-specific 2021 rate, then multiplied by the age-specific calibration/old-age factor. The default horizon is 50 calendar years after 2021 and is shared by all cohorts and interventions. Beyond it, time factors stop changing, but mortality can continue increasing with age. The horizon is an assumption and is varied in sensitivity analysis.

COVID uses a separate multiplier:

```text
residual + (1 - residual) × 2 ** (-(year - 2021) / half_life)
```

The default half-life of 3 years and residual 10% are illustrative, not estimated. Holding 2021 constant and complete exclusion are available as explicit counterfactuals. The baseline year includes observed COVID; starting a cohort later moves it along the selected path. Indirect pandemic effects in other 2021 causes and future outbreaks are not modeled.

## Interventions

The two-stage fractional annual reduction applies its first reduction in the stated start year. The floor limits only the slow-stage multiplier. At the fast boundary the accumulated slow multiplier is retained, then fast reductions compound without a floor:

```text
multiplier = max(slow_floor, (1 - slow_rate) ** slow_years)
             * (1 - fast_rate) ** fast_years
```

`slow_years` stops accumulating just before fast begins. A zero fast rate preserves that slow multiplier; a 100% fast reduction makes the cause's hazard exactly zero from the fast start year. A manually specified fast rate of 10% leaves `0.9 ** 10 ≈ 0.349` of the counterfactual risk after ten fast reductions, on top of the slow effect. CSV column `Floor` means slow floor; its name is retained for compatibility.

### Fast defaults compensate exponential ageing

For a positive log-slope `b` in the active exponential tail, the UI initializes:

```text
suggested_fast_rate = 1 - 0.9 * exp(-b)
one_year_net_factor = exp(b) * (1 - suggested_fast_rate) = 0.9
```

This is multiplicative compensation, not addition of percentage points. `b` includes the selected slope multiplier and any custom curve. Dementia's default `b = 0.1980141115` gives a fast reduction of `0.2616775511`. The guarantee describes annual hazard changes after 94 during the fast stage with calendar factors fixed. Continuing secular trends, COVID transitions and source interpolation before 94 can alter the realized change. Plateau, sigmoid and declining families retain a 10% treatment default; common Gompertz mode compensates its shared exponential slope for every cause. Flat and common Kannisto modes have no positive exponential tails for this purpose.

Defaults update when an active shape changes only while the value remains automatic. Manual edits and imports retain their literal treatment reduction. An explicit reset applies the current suggestions. Untracked values from older sessions are preserved until reset. The bulk-selection button adds positive-slope exponential causes under the active model, preserving previous selections. Neither action changes CSV units or the intervention equations; assumptions JSON records both actual interventions and the suggested-rate policy.

Evaluation uses `log1p(-rate)` and a separate branch for a 100% annual reduction. The mortality calculation retains the log multiplier through its combination with ageing, avoiding premature underflow of tiny fast multipliers. All stages act on top of the chosen counterfactual; an intervention should represent progress additional to any historical extrapolation.

Interventions are still uniform by age, without treatment uptake, adherence, disease states or correlated individual frailty. Changes in one cause do not directly modify another cause's hazard. Competing deaths nevertheless remain coupled through total survival.

## Survival and numerical horizon

Within each one-year interval, summed cause rates are interpreted as constant hazards `h`:

```text
S(a + 1) = S(a) × exp(-h)
life-years in interval = S(a) × (1 - exp(-h)) / h
```

For `h=0`, the contribution is `S(a)`. `expm1` provides stable integration when hazards are tiny. Percentile crossing times use the inverse exponential within the year. The plotted curve samples these exponentials quarterly and retains the whole numerical horizon. Its initial zoom covers at least 99% of deaths when that percentile is reached; a range slider exposes the rest of the horizon.

The integration ends at a configurable age `H` (default 2,000, allowed range 100–10,000). This supports radical life-extension scenarios while bounding computation; it is not a maximum age at death. We retain:

- the restricted expected remaining years before `H`;
- survival `S(H)`;
- percentiles reached before `H`;
- an upper bound on the remaining expected life-years.

Old-age hazards, cause scaling, COVID decay and intervention effects are combined in logarithms before exponentiation. This avoids overflowing a Gompertz baseline before a small multiplier is applied, or forming `inf * 0` for excluded causes. A final hazard beyond floating-point range is represented as infinity and integrated using its zero-survival, zero-life-years limit; no artificial maximum hazard is imposed. Survival at the exact start of such an interval is retained explicitly. These numerical protections do not make extrapolations to remote ages empirically validated.

For tail analysis, first advance to an anchor after time trends and all fast-stage transitions have finished, explicitly checking any intervening annual hazards. Thereafter each nonzero cause is a geometric sequence multiplied by its age curve. COVID fade is split into residual and decaying components. Declining curves with a residual similarly split into constant and decreasing terms; these splits combine when a COVID cause is assigned a declining curve.

For plateau, sigmoid and common Kannisto ageing, the geometric log-growth is `log(1 - fast_rate)`, with an additional `-log(2)/half_life` for a fading COVID component. Exponent/Gompertz adds its positive ageing slope; a declining term subtracts its slope. Sigmoid hazards are nondecreasing and bounded between their current value and their individual asymptote. Non-decaying terms provide a conservative positive future hazard bound `h_min`, combined with the minimum across any years before the anchor. Thus:

```text
remaining integral <= S(H) / h_min
```

This bound requires no frozen tail assumption. A finite mean is reported only when the omitted contribution is at most 0.001 years; otherwise it remains `None` / “Unresolved”.

If every remaining component decays geometrically, or all have been eliminated, total future cumulative hazard is finite. Survival then has a positive limit and the full mean is analytically **infinite**, regardless of the integration horizon. This classification also accounts for Gompertz growth: fast improvements need to outpace ageing for all surviving terms. Zero numerical hazard at a distant age alone is not used as proof.

For a decreasing term `h(n) = c * exp(g*n)`, the cumulative hazard is `c / (1 - exp(g))`. Plateau, exponent and declining terms use this geometric sum directly. Sigmoids use their own asymptote/current ratio as an upper multiplier; common Kannisto uses its 1/year asymptote. Adding the pre-anchor hazard gives a conservative remaining cumulative-hazard bound `U`. Then `S(infinity) >= S(H) * exp(-U)`. Percentiles are marked infinite only when this lower survival bound proves they cannot be reached. A bound can underflow to zero while the structural proof of an infinite mean remains valid. Restricted life-years remain finite and comparable at a common horizon in all cases.

Sensitivity analysis compares deterministic alternative settings, using the same settings for the paired baseline and intervention and the same horizon for restricted means. These ranges are not probability intervals. No claim is made that a narrow sensitivity range demonstrates forecast accuracy.
