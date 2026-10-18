# Lifespan Simulator

A Streamlit scenario explorer for **male mortality in Spain**, using cause-specific rates from 2011–2021 and a checked-in Eurostat reference snapshot for 2021.

The app estimates survival conditional on having reached the selected initial age. It is a population scenario model, not a personalized or validated long-term forecast.

## Run and check

```bash
uv sync --frozen
uv run streamlit run app.py
```

Or launch with `./run_app.sh`. The launcher enables file polling and automatic reruns while editing; additional Streamlit arguments can be passed to it. Restart a long-running server after code changes if it continues to display an older model. Tests and the running app do not need network access after installation.

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

## What is included

- **Simulator:** selectable cohort starting year and age; historical trends; a separate COVID path; two-stage interventions; survival curves, life expectancy and death-age percentiles.
- **Cause trajectories:** individual exponent, sigmoid, plateau or declining tails for 68 causes; inspect fitted curves, selection scores and fallback assumptions, and export the full fit table. Dementia and Alzheimer disease are combined.
- **Reference checks:** compare the source cause totals with official all-cause rates, and reconstruct period remaining life expectancy at 35, 65, 85 and 94.
- **Sensitivity analysis:** change COVID, old-age mortality, trend assumptions, calibration and intervention parameters one at a time; export the paired baseline/scenario results as CSV.
- **Historical explorer:** inspect the original time series and fitted mean confidence bands, including pandemic years.

## Default assumptions

| Setting | Default |
|---|---|
| Cohort | Age 35 in 2021; starting year is configurable |
| Baseline calibration | Match Eurostat single-age all-cause Mx at ages 35–94, preserving source cause shares |
| Old-age mortality | Individual cause trajectories, fitted to closed groups at ages 70–94 in 2011–2019 and spliced at 94; common Kannisto, Gompertz and flat curves remain available for comparison |
| Declining tails | Exponential decline toward zero; an optional residual share tests continued background risk |
| Historical trends | Significant linear cause-age slopes from 2011–2019; COVID excluded |
| Trend bounds | 15% to 2× the age-specific baseline; extrapolate for 50 years after 2021, then hold time factors fixed |
| COVID | Fade from its 2021 rate with a 3-year half-life toward a 10% residual share |
| Intervention | Slow start 2040, reduction 1.5%/year, slow floor 1%; fast start 2070, no fast floor; exponential causes default to age-growth compensation plus 10% net reduction/year, other families to a 10% treatment reduction |
| Numerical horizon | Age 2,000, editable from 100 to 10,000; not a maximum human lifespan |

COVID fade, trend bounds/horizon and intervention values are **scenario assumptions**, not estimates of medical progress. The non-COVID 2021 baseline may retain indirect pandemic effects. Interventions apply additional reductions relative to the chosen baseline; avoid counting the same progress in both the trend and the intervention.

**All 95+ observations are excluded** from the simulator, age-curve fits, historical trends, calibration and reference comparisons, including the legacy modes. The last closed cause group is 90–94. The raw source files are unchanged. Beyond 94, cause rates are extrapolated without renormalizing their sum to a common age curve.

The first-version selector prefers simple forms on near ties and labels weak evidence explicitly. Fourteen sparse or unavailable series use plateau fallbacks; this is not evidence of a measured biological plateau. Each cause's family and parameters can be changed under **Override individual cause trajectories**. Overrides appear in the trajectory table and assumptions JSON with their validation score cleared. Details and limitations are in [the model description](docs/model.md).

Dementia is an explicit scenario choice: its default is an exponent fitted to ages **70–89**, with a log-slope of about 0.198/year (doubling in 3.50 years). The previous sigmoid fitted to 70–94 remains available through **Dementia age trajectory** inside the override expander. Both curves appear in the inspection chart. The primary slope excludes 90–94; the common age-94 anchor still uses the baseline source level and optional calibration. No automatic selection score is assigned to the chosen exponent.

For an active exponential log-slope `b`, the suggested fast reduction is `1 - 0.9 * exp(-b)`, so age growth and treatment together multiply the hazard by 0.9 each year. Dementia therefore defaults to **26.167755%**, rather than a raw 10% reduction. The net target applies after 94 with fixed calendar factors; it does not compensate ongoing time trends or changing COVID levels. Non-exponential families retain a raw 10% default. Common Gompertz mode uses its shared slope for all causes.

**Select all exponentially growing causes** adds the active positive-slope exponent causes to the selection (38 with current defaults), preserving existing selections. Automatic fast values follow trajectory changes until edited. Manual values and CSV imports are preserved; **Reset selected fast rates to defaults** explicitly applies the current suggestions to selected causes. Existing sessions without provenance tracking retain their saved values until reset. Exports store actual treatment reductions with the same CSV semantics as before.

## Reading the results

- **Mean age at death** is initial age plus expected remaining life. It is not life expectancy at birth.
- **p75** is the age by which 75% of the modeled cohort has died. Percentiles are not confidence intervals for the forecast.
- **Unresolved** means the simulation completed, but surviving beyond the numerical horizon prevents a sufficiently precise full mean. The median can still be known because it only requires half of the cohort to have died. The app shows expected remaining years lived *before the horizon*, remaining survival, an upper bound on omitted life-years, and numerical lower/upper bounds on mean age at death. An unavailable mean has no gain indicator. Increase the horizon up to age 10,000 for longer-lived scenarios; especially small residual hazards can still leave the mean unresolved.
- Mean-age bounds are obtained by adding the initial age to the restricted remaining years, then adding the omitted life-years bound for the upper endpoint. A blank upper mean bound means no finite bound was established, not proof of an infinite mean. These bounds describe numerical integration under the selected model, not uncertainty about future medical progress; no constant-risk tail is assumed.
- **∞** means that the model analytically retains positive survival forever, making the full mean infinite. This can occur when fast improvements suppress every remaining mortality term indefinitely. Increasing the horizon cannot make that mean finite. Restricted life-years, reached percentiles and a lower bound on limiting survival remain available. This is a consequence of the selected assumptions, not evidence of real-world immortality.
- A finite resolved mean omits at most **0.001 years** under the selected model. This is numerical error control, not forecast accuracy.
- **Beyond horizon** means a percentile was not reached within the calculation. A percentile is **∞** only when the limiting-survival bound proves that it can never be reached.
- The survival plot retains the entire computed horizon, with a range slider for exploring distant ages. Its initial view covers at least 99% of deaths if that percentile was reached, avoiding a mostly empty plot out to age 2,000.
- Sensitivity rows are deterministic alternatives, not samples from a probability distribution or a confidence band.

## Reference data and reproducibility

The source `mortality.csv` is retained unchanged. The original extraction metadata are not available. New reference data are sourced from Eurostat for Spain, males, 2021:

- [`demo_mlifetable`](https://ec.europa.eu/eurostat/databrowser/view/demo_mlifetable/default/table): Mx (deaths per person-year) and ex (remaining years).
- [`hlth_cd_acdr2`](https://ec.europa.eu/eurostat/databrowser/view/hlth_cd_acdr2/default/table): all-cause deaths per 100,000, matching the source age groups.

`data/reference_metadata.json` records exact requests, retrieval date, source update timestamps, flags, units and SHA-256 checksums. The stored 95+ group is an open-group average and is excluded at runtime. Calibration does not explain or remove discrepancies from the source audit.

To explicitly refresh the snapshot (requires network):

```bash
uv run python scripts/refresh_reference_data.py
uv run pytest -q
```

Review changes to the reference files and tests before accepting an updated snapshot. The app never downloads or silently refreshes data.

See [model equations and limitations](docs/model.md) and [validation results](docs/validation.md).

## Intervention CSV

```text
Cause,Slow start year,Slow annual reduction,Fast start year,Fast annual reduction,Floor
```

Values are fractions: `0.015` means 1.5% per year. The `Floor` CSV column is retained for compatibility and now limits **only the slow-stage multiplier**. The accumulated slow effect is fixed at the fast boundary; fast improvements then compound without a floor on top of it. A fast rate of `0.1` leaves 90% each year (34.9% after ten reductions); `1` eliminates this cause from the fast start year, and `0` preserves the completed slow effect. Both stages multiply the cause's counterfactual rate after time trends and the chosen COVID path. The UI calls this parameter **Slow floor**, and the assumptions JSON records its scope. Existing CSVs therefore have updated floor semantics. Years must be integers in 2021–2120; fast start must not precede slow start; rates must be in [0, 1]; slow floor in [1e-12, 1]. Invalid and non-finite values are rejected.

New exports use `Dementia including Alzheimer disease`. Two legacy dementia/Alzheimer rows are merged only when both have identical settings. A single legacy row or conflicting settings require an explicit combined row, avoiding an unintended expansion of the intervention.

## Code layout

- `app.py`: entry point, cached inputs and tabs.
- `data.py`: source normalization, validation and reference loading.
- `config.py`: validated, immutable projection assumptions.
- `tails.py`: compact cause-curve families, pre-pandemic shape selection and fit diagnostics.
- `mortality.py`: reusable age/cause surface, calibration, pandemic path and old-age curves.
- `interventions.py` / `scenario_io.py`: intervention paths and CSV interchange.
- `simulation.py`: analytical within-year survival, life-year integration and numerical remainder bounds.
- `analysis.py`: reference comparisons and paired sensitivity experiments.
- `trends.py`: historical regressions and pre-pandemic trend preparation.
- `charts.py` / `ui.py`: presentation and Streamlit widget state.
- `tests/`: numerical, data, input-validation and Streamlit interaction tests.

## Python API

Prepare one mortality surface and reuse it for comparisons:

```python
from lifespan_simulator.config import ProjectionConfig
from lifespan_simulator.data import (
    load_mortality_data,
    get_simulator_frame,
    load_official_life_table,
)
from lifespan_simulator.mortality import MortalityModel
from lifespan_simulator.simulation import simulate_survival
from lifespan_simulator.tails import fit_cause_tails
from lifespan_simulator.trends import build_trend_model

raw = load_mortality_data()
frame = get_simulator_frame(raw)
trend = build_trend_model(raw, frame)
tails = fit_cause_tails(raw, tuple(sorted(frame.cause_code.unique())))
model = MortalityModel(frame, trend.slope_table, load_official_life_table(), tails)
result = simulate_survival(model, initial_age=35, config=ProjectionConfig(start_year=2026))
```

Projection functions accept `MortalityModel` and `ProjectionConfig`. Full mean fields are finite numbers when numerically converged, `None` when unresolved, or `math.inf` when infinity has been established analytically. `converged` identifies finite numerical convergence; `mean_resolved` includes analytically infinite means; `infinite_mean` distinguishes them. Use `restricted_remaining_life_expectancy`, `survival_at_end`, `remaining_expectation_bound` and `survival_forever_lower_bound` for diagnostics. Export the UI's assumptions JSON alongside sensitivity CSVs to record the chosen settings.

Supply `tail_fits` when constructing the default individual model. Without pre-pandemic history, a 2021-only frame produces explicitly marked plateau fallbacks.
