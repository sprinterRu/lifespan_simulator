# Model and assumptions

## Mortality surface

Let `r_c(a)` denote the interpolated 2021 rate for cause `c`, per 100,000. Closed five-year observations are interpolated across their age midpoints. All 95+ observations are excluded from every projection mode, calibration, trend fit and reference comparison. Raw snapshots remain unchanged. The last closed group's midpoint is 92 (90–94); source shares and time-trend slopes are held at that group beyond 92. Official all-cause calibration still varies through age 94. Direct causes receive their own age multipliers after 94; three reporting categories follow identified cause groups.

The disjoint underlying-cause codes `F01_F03` and `G30` are summed into `F01_F03_G30` before fitting either age shapes or calendar trends. The simulator has 65 causes. Cervical (`C53`), other uterine (`C54_C55`) and ovarian (`C56`) cancers are excluded from this male model and UI because all their source rates are zero. Male breast cancer and rare congenital/perinatal categories remain. Raw data are unchanged. This sums recorded causes of death; it does not infer incidence, prevalence or a clinical dementia diagnosis. Missing component cells are rejected rather than filled with zeros. The historical explorer preserves the two original series, excluding 95+.

For calibrated projections, the total baseline rate is `100000 × Mx(a)` for ages 35–94, from the official Eurostat life table. Every cause receives the same age-specific factor:

```text
calibration_factor(a) = official_total_rate(a) / sum_c r_c(a)
```

This preserves the source cause shares. It is an explicit modeling choice, not evidence that every cause-specific rate is correct. The source audit compares unmodified five-year rates against the *separate* Eurostat cause-of-death all-cause series. Life-table and cause-of-death production methods differ, so those two reference series are not forced to agree.

Uncalibrated mode uses the source totals instead. It is included for comparison and sensitivity analysis.

## Older ages

The default family for each cause is a reviewed scenario prior in `cause_models.py`. Its description, rationale and provisional status accompany the curve in the UI and exports. For `x = age - 94 >= 0`:

| Family | Multiplier | Parameters |
|---|---|---|
| Slowing growth (P / `power`) | `(1 + x/tau) ** (b*tau)` | Initial log-slope `b >= 0`, slowdown `tau > 0` |
| Bounded growth (B / `bounded`) | `exp(b*tau*(1-exp(-x/tau)))` | Initial log-slope `b >= 0`, slowdown `tau > 0` |
| Flat (F / `plateau`) | `1` | None |
| Decline with residual (D / `decline`) | `r + (1-r)*exp(-d*x)` | Decay parameter `d >= 0`, residual `0 <= r < 1` |
| Exponent (comparison) | `exp(b*x)` | Nonnegative slope `b` |
| Sigmoid (comparison) | `R / (1 + (R-1)*exp(-b*x))` | Slope `b`, ratio `R >= 1` |

Every multiplier equals one at the splice. Slowing growth has no ceiling; bounded growth approaches `exp(b*tau)` times the anchor. The defaults are **tau = 20 years** and **r = 0.25**, explicitly assumed rather than fitted. Slowdown is editable per cause; the declining residual is shared and editable in Mortality assumptions. No hazard ceiling of 1/year is imposed by these families. A flat positive rate continues accumulating deaths.

There are 25 power, 24 bounded, 10 flat, 3 decline and 3 linked categories. The three zero-only cancer categories are removed instead of retaining zero curves. Independent causes are not renormalized to an external total after 94. Future treatment remains separate from the baseline age shapes and can reduce a declining residual further.

### Parameter estimation and uncertainty

Only 2011–2019 closed groups below 95 inform age slopes. Baseline levels remain anchored to 2021. The midpoint representation approximates group-average rates, not single-age observations or exposure-weighted estimation.

- Growth slopes use a log-linear fit to the mean rates at midpoints 82, 87 and 92 (80–94), clipped to [0, 0.4].
- Support requires at least five complete positive years, positive means, mean rate at least 5 per 100,000 over the last two groups, and at least 80% positive values in each of the five 70–94 groups. These are pragmatic safeguards, not uncertainty estimates.
- For sparse growth causes with positive pooled means, use 25% own mean-profile slope plus 75% median of supported growing causes in the same registry group. Zeros remain in the mean; no pseudocount is added. Without a usable own slope, use the group median. Without a supported group, use the explicitly assumed 0.05/year slope.
- All three COVID categories borrow the same supported infection-group median. No pre-pandemic COVID fit is claimed. Borrowing pools use the full available history, so requesting a subset of causes cannot change their parameters.
- Declining causes use `g = max(0, -log(mean_90_94 / mean_85_89)/5)`, capped at 0.4, and `d = g/(1-0.25)`. This matches the initial decline with the default residual. Changing the residual also changes the initial slope; the stored decay parameter stays fixed. With unavailable positive terminal means the decay is zero and explicitly flagged.
- Flat and linked families have no independently fitted age slope. A zero baseline cell stays zero unless the ordinary calendar model supplies risk; no positive baseline rate is invented by the age fitter.

The family, 20-year slowdown, pooling weights, residuals and fallback slope are assumptions. Default families receive no validation score or claimed selection probability. Sparse and heterogeneous causes are labelled provisional. Death counts/exposures are absent, and survivor selection, cohort history and classification can affect age profiles. Life-extending interventions may change those patterns.

The earlier descriptive exponent/sigmoid/plateau selector remains available for unregistered synthetic/API causes and the dementia comparison. Its blocked-age scores concern prediction below 95, not validation of remote extrapolations.

### Linked reporting categories (L)

`R96-R99` and `R_OTH` follow identified natural causes. `Y10-Y34_Y872` follows identified external causes. Donor groups exclude all three reporting codes and any actively linked custom curves, so the dependency graph has no cycles.

At each age through the boundary, derive `rho_i(a) = raw_i(a) / sum(raw_donors(a))` from the observed 2021 cause rates. These ratios are fixed in calendar time and held at the last closed midpoint after age 92, as are the other source shares. Then, in individual mode:

```text
linked_rate_i(a, year) = rho_i(a) * sum(projected_donor_rates(a, year))
                         * explicit_intervention_i(year)
```

Donor rates already include age shape, calibration, calendar trends, COVID and donor interventions. The linked category's own calendar trend is replaced by donor trends. Applying this rule throughout the observed ages preserves the 2021 level and avoids a discontinuity at 94 in future calendar years. A direct intervention on a linked category is an additional multiplier, explained in the UI. Removing a donor reduces its linked reporting share but leaves other direct causes unchanged; no hazard is redistributed into other diseases. A positive reporting rate with no positive donor reference is rejected.

### Dementia

Combined dementia uses slowing growth with initial log-slope **0.1497565669/year**, estimated over 80–94, and assumed slowdown **20 years**. The earlier 70–94 sigmoid (slope 0.2167405741, age-94 asymptote ratio 1.528725466) remains a selectable comparison. Its log-RMSE 0.1165226056 measures blocked-age predictions below 95. Both are exported, and neither models disease incidence or prevalence.

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

For each direct cause (and each cause in common-curve comparison modes), the source rate is advanced by `slope × elapsed`, where:

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

This is multiplicative compensation, not addition of percentage points. `b` includes the selected slope multiplier and any custom curve. The guarantee describes annual hazard changes after 94 during the fast stage with calendar factors fixed. Continuing secular trends, COVID transitions and source interpolation before 94 can alter the realized change. Power, bounded, plateau, sigmoid, declining and linked families retain a raw 10% treatment default; common Gompertz mode compensates its shared exponential slope for every cause. Flat and common Kannisto modes have no positive exponential tails for this purpose.

Defaults update when an active shape changes only while the value remains automatic. Manual edits and imports retain their literal treatment reduction. An explicit reset applies the current suggestions. Untracked values from older sessions are preserved until reset. The bulk-selection button adds all direct increasing causes (49 under the reviewed defaults), preserving previous selections. Neither action changes CSV units or the intervention equations; assumptions JSON records both actual interventions and the suggested-rate policy.

Evaluation uses `log1p(-rate)` and a separate branch for a 100% annual reduction. The mortality calculation retains the log multiplier through its combination with ageing, avoiding premature underflow of tiny fast multipliers. All stages act on top of the chosen counterfactual; an intervention should represent progress additional to any historical extrapolation.

Interventions are still uniform by age, without treatment uptake, adherence, disease states or correlated individual frailty. Changes in one direct cause do not modify other direct causes; linked reporting categories inherit donor changes. Competing deaths nevertheless remain coupled through total survival.

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

For power, bounded, plateau, sigmoid and common Kannisto ageing, the geometric log-growth is `log(1 - fast_rate)`, with an additional `-log(2)/half_life` for a fading COVID component. Exponent/Gompertz adds its positive ageing slope; a declining term subtracts its slope. Bounded and sigmoid hazards lie between their current rate and their asymptote. Power hazards are nondecreasing with a polynomial upper factor. Non-decaying terms provide a conservative positive future hazard bound `h_min`, combined with the minimum across any years before the anchor. Thus:

```text
remaining integral <= S(H) / h_min
```

This bound requires no frozen tail assumption. A finite mean is reported only when the omitted contribution is at most 0.001 years; otherwise it remains `None` / “Unresolved”.

If every remaining component has negative geometric log-growth (possibly multiplied by a polynomial), or all have been eliminated, total future cumulative hazard is finite. Survival then has a positive limit and the full mean is analytically **infinite**, regardless of the integration horizon. This classification also accounts for Gompertz growth: fast improvements need to outpace ageing for all surviving terms. Zero numerical hazard at a distant age alone is not used as proof.

For a decreasing term `h(n) = c * exp(g*n)`, the cumulative hazard is `c / (1 - exp(g))`. Plateau, exponent and declining terms use this geometric sum directly. Sigmoids use their own asymptote/current ratio as an upper multiplier; common Kannisto uses its 1/year asymptote. Adding the pre-anchor hazard gives a conservative remaining cumulative-hazard bound `U`. Then `S(infinity) >= S(H) * exp(-U)`. Percentiles are marked infinite only when this lower survival bound proves they cannot be reached. A bound can underflow to zero while the structural proof of an infinite mean remains valid. Restricted life-years remain finite and comparable at a common horizon in all cases.

For a power tail, the multiplier relative to a future anchor is `(1+n/s)**p`, where `p=b*tau` and `s=tau+anchor_age-94` (including the slope multiplier in `b`). With geometric growth `g<0`, choose `epsilon=min(-g/2,p/s)` and maximize `p*log(1+n/s)-epsilon*n` over `n>=0`. Its maximum occurs at `max(0,p/epsilon-s)`. This gives a finite exponential envelope with remaining growth `g+epsilon<0`, even if the treated hazard initially increases. The envelope is summed geometrically. A bounded curve uses its exact asymptote/current ratio. Linked categories copy donor components and add any own intervention multiplier and decay, so their bounds also include donor COVID and treatment effects.

Sensitivity analysis compares deterministic alternative settings, using the same settings for the paired baseline and intervention and the same horizon for restricted means. These ranges are not probability intervals. No claim is made that a narrow sensitivity range demonstrates forecast accuracy.
