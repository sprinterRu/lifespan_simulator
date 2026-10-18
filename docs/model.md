# Model and assumptions

## Mortality surface

Let `r_c(a)` denote the interpolated 2021 rate for cause `c`, per 100,000. Five-year observations are interpolated across their age midpoints. The 95+ observation is placed at the boundary 95 so there is no interpolation jump caused by a fictitious midpoint at 97. Cause shares and source trend slopes are held fixed after age 95.

For calibrated projections, the total baseline rate is `100000 × Mx(a)` for ages 35–94, from the official Eurostat life table. Every cause receives the same age-specific factor:

```text
calibration_factor(a) = official_total_rate(a) / sum_c r_c(a)
```

This preserves the source cause shares. It is an explicit modeling choice, not evidence that every cause-specific rate is correct. The source audit compares unmodified five-year rates against the *separate* Eurostat cause-of-death all-cause series. Life-table and cause-of-death production methods differ, so those two reference series are not forced to agree.

Uncalibrated mode uses the source totals instead. It is included for comparison and sensitivity analysis.

## Older ages

The official reference contains single-age observations only through 94. The primary model fits a non-negative slope to `log(Mx / (1 - Mx))` at ages 80–94 and anchors the extrapolation at the observed age-94 rate:

```text
mu(a) = 1 / (1 + (1 / mu(94) - 1) × exp(-b × (a - 94)))
```

The asymptote is 1 per year. The fit is unweighted least squares, **not** the exposure-weighted likelihood estimation used in the [Human Mortality Database methods protocol](https://www.mortality.org/File/GetDocument/Public/Docs/MethodsProtocolV6.pdf). We do not have the requisite observed deaths and exposures in the source CSV.

Alternative models are:

- Gompertz: fit a non-negative slope to `log(Mx)` over 80–94, then `mu(a) = mu(94) × exp(b × (a-94))`.
- Flat 95+: keep the open-group average from age 95 onward, as a diagnostic comparator.

The same procedures use source totals when calibration is disabled. Projected totals are allocated to the cause shares before cause-specific trend, COVID and intervention adjustments. Therefore, interventions can lower the old-age asymptote: this does not impose a biological lifespan limit. Extrapolation into extreme ages remains weakly supported, even with a better baseline curve.

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

`slow_years` stops accumulating just before fast begins. A zero fast rate preserves that slow multiplier; a 100% fast reduction makes the cause's hazard exactly zero from the fast start year. The default fast rate remains 10% annually, leaving `0.9 ** 10 ≈ 0.349` after ten fast reductions, on top of the slow effect. CSV column `Floor` now means slow floor; its name is retained for compatibility.

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

For ages at least 95, cause shares/slopes are fixed. Tail analysis first advances to an anchor after time trends and all fast-stage transitions have finished, explicitly checking any intervening annual hazards. Thereafter each nonzero cause is a geometric sequence multiplied by the age curve. COVID fade is split into residual and decaying components.

For flat/Kannisto ageing the geometric log-growth is `log(1 - fast_rate)`, with an additional `-log(2)/half_life` for a fading COVID component. Gompertz adds its positive ageing slope. Non-decaying components provide a conservative positive future hazard bound `h_min`, combined with the minimum across any years before the anchor. Thus:

```text
remaining integral <= S(H) / h_min
```

This bound requires no frozen tail assumption. A finite mean is reported only when the omitted contribution is at most 0.001 years; otherwise it remains `None` / “Unresolved”.

If every remaining component decays geometrically, or all have been eliminated, total future cumulative hazard is finite. Survival then has a positive limit and the full mean is analytically **infinite**, regardless of the integration horizon. This classification also accounts for Gompertz growth: fast improvements need to outpace ageing for all surviving terms. Zero numerical hazard at a distant age alone is not used as proof.

For a decreasing term `h(n) = c * exp(g*n)`, the cumulative hazard is `c / (1 - exp(g))`. Flat and Gompertz tails use this geometric sum directly; Kannisto uses its 1/year asymptote to bound future ageing from above. Adding the pre-anchor hazard gives a conservative remaining cumulative-hazard bound `U`. Then `S(infinity) >= S(H) * exp(-U)`. Percentiles are marked infinite only when this lower survival bound proves they cannot be reached. A bound can underflow to zero while the structural proof of an infinite mean remains valid. Restricted life-years remain finite and comparable at a common horizon in all cases.

Sensitivity analysis compares deterministic alternative settings, using the same settings for the paired baseline and intervention and the same horizon for restricted means. These ranges are not probability intervals. No claim is made that a narrow sensitivity range demonstrates forecast accuracy.
