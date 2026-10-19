# Validation snapshot

Reference population: males in Spain, 2021. See `data/reference_metadata.json` for requests, retrieval time, source revisions, units and hashes. All 95+ observations are excluded. These checks validate the implementation and reconstruction, not extreme-age predictions.

## Source totals and cause coverage

The simulator contains **65 causes** across 12 closed groups (35–39 through 90–94), after excluding overlapping aggregates, merging dementia/Alzheimer disease, and removing cervical, other uterine and ovarian cancers. The three removed categories have zero rates throughout the source, so their removal changes no source total. Male breast cancer, congenital conditions and perinatal sequelae remain.

The largest absolute source/official discrepancy remains **−3.51 per 100,000**, at 65–69. The largest relative discrepancy remains **−0.587%**, at 50–54. The audit precedes calibration, and raw snapshots are unchanged.

## Period life expectancy

Disable calendar trends, retain 2021 COVID and apply no interventions. Individual curves use reviewed families, a 20-year slowdown and a 25% declining residual.

| Starting age | Official remaining years | Source + flat | Source + Kannisto | Calibrated + Kannisto | Calibrated + individual | Individual error |
|---|---:|---:|---:|---:|---:|---:|
| 35 | 45.9 | 46.004 | 45.878 | 45.896 | 45.886 | −0.014 |
| 65 | 18.9 | 19.110 | 18.969 | 18.897 | 18.886 | −0.014 |
| 85 | 6.1 | 6.703 | 6.410 | 6.086 | 6.064 | −0.036 |
| 94 | 3.1 | 4.987 | 3.848 | 2.975 | 2.879 | −0.221 |

Published ex is rounded to 0.1 years. The regression tolerances are 0.1 years before 94 and 0.3 at 94, not claims of scientific accuracy. No open-age observation or life-expectancy statistic is used to tune the new tails.

## Reviewed families and parameter provenance

The registry covers every retained cause exactly once: **25 slowing-growth, 24 bounded-growth, 10 flat, 3 declining-with-residual and 3 linked**. Every entry has a description and rationale. Default family choices have no statistical selection score. The 20-year slowdown, 25% residual and sparse-slope pooling weights are explicit assumptions.

Dementia's initial log-slope is 0.1497565669/year, estimated from 80–94. The earlier sigmoid remains available as a comparison (slope 0.2167405741, plateau ratio 1.528725466, blocked-age log-RMSE 0.1165226056). Its score only concerns predictions below 95. The three COVID codes borrow the same supported infection-group slope; no pre-pandemic COVID fit is claimed.

Linked categories preserve baseline observations and follow donor age curves, calendar trends, COVID and treatment effects. They are excluded from their own donor groups. A donor reduction never increases another direct cause's hazard. An intervention on a linked code adds an explicitly documented extra multiplier.

## Illustrative scenario outputs

Age 35 in 2021, calibrated, 2011–2019 trends with a 50-year horizon, fading COVID, default reviewed tails. Interventions use slow 1.5% from 2040, a 1% slow floor, and raw 10% fast reduction with no fast floor.

| Intervention | Fast start | Mean age | Median | p99 |
|---|---:|---:|---:|---:|
| None | — | 84.651 | 87.703 | 101.870 |
| All 49 direct growing causes | 2070 | Unresolved at horizon 2,000 | 92.389 | 611.254 |
| All 65 causes | 2070 | Infinite | 93.402 | Infinite |
| All 65 causes | 2080 | Infinite | 90.840 | 135.520 |

The all-cause 2070 scenario has a proven limiting-survival lower bound of 9.346%; the 2080 scenario's bound is 0.612%. These are consequences of perpetual treatment assumptions, not estimates of achievable medicine. Exponential suppression eventually dominates polynomial growth. The growing-causes-only scenario retains untreated positive risks; its unresolved mean reflects the numerical tolerance, not a proof of an infinite mean.

Manual and imported treatment rates are retained literally. The new non-exponential defaults use a raw 10% reduction; exponential overrides and the common Gompertz comparator retain age-compensated suggestions. Older scenario outputs based on the previous 38 exponential tails are no longer current defaults.

## Automated and interface checks

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

The suite covers analytical constant-hazard integrals and percentiles, monotone survival, probability conservation, no-op interventions, chart/simulation consistency, intervention boundary years, tiny and 100% reductions, remote-age numerical stability, reference checksums, calibration and CSV round trips.

Checks for this revision cover all 65 registry entries; removal of only zero-only cancer categories; unchanged source totals; descriptions and formulas in Streamlit; editable slowdown and residuals; sparse pooling independent of requested subsets; shared COVID shapes; pre-pandemic-only estimation; continuity, initial derivatives and remote limits of both new growth families; and preservation of manual/imported treatment settings.

Future-bound checks compare computed envelopes with long direct hazard sums for polynomial, bounded, declining and legacy curves, including initially increasing hazards under weak treatment. Linked-code tests cover donor treatment, COVID decay, baseline preservation, continuity, extra direct multipliers and conservative survival bounds. The live local app is also inspected after restart.

## Remaining scientific work

- Death counts and population exposures for weighted fits and sampling uncertainty.
- Independent evidence for extreme-age shape, slowdown timescale and residual levels.
- Separation of survivor selection, cohort, exposure and coding effects.
- Disease-state models and age-dependent treatment uptake/effectiveness.
- A probabilistic uncertainty model; current sensitivity comparisons are not confidence intervals.
