# Lifespan Simulator

A Streamlit scenario explorer for **male mortality in Spain**, using cause-specific rates from 2011–2021 and a checked-in Eurostat reference snapshot for 2021.

The app estimates survival conditional on having reached the selected initial age. It is a population scenario model, not a personalized or validated long-term forecast.

## Run and check

```bash
uv sync --frozen
uv run streamlit run app.py
```

Or launch with `./run_app.sh`. Tests and the running app do not need network access after installation.

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

## What is included

- **Simulator:** selectable cohort starting year and age; historical trends; a separate COVID path; two-stage interventions; survival curves, life expectancy and death-age percentiles.
- **Reference checks:** compare the source cause totals with official all-cause rates, and reconstruct period remaining life expectancy at 35, 65, 85 and 95.
- **Sensitivity analysis:** change COVID, old-age mortality, trend assumptions, calibration and intervention parameters one at a time; export the paired baseline/scenario results as CSV.
- **Historical explorer:** inspect the original time series and fitted mean confidence bands, including pandemic years.

## Default assumptions

| Setting | Default |
|---|---|
| Cohort | Age 35 in 2021; starting year is configurable |
| Baseline calibration | Match Eurostat single-age all-cause Mx at ages 35–94, preserving source cause shares |
| Old-age mortality | Kannisto curve fitted at ages 80–94, anchored at 94; compare Gompertz and flat 95+ alternatives |
| Historical trends | Significant linear cause-age slopes from 2011–2019; COVID excluded |
| Trend bounds | 15% to 2× the age-specific baseline; extrapolate for 50 years after 2021, then hold time factors fixed |
| COVID | Fade from its 2021 rate with a 3-year half-life toward a 10% residual share |
| Intervention | Slow start 2040, reduction 1.5%/year; fast start 2080, reduction 10%/year; residual multiplier 1% |
| Numerical horizon | Age 200, selectable up to 500; not a maximum human lifespan |

COVID fade, trend bounds/horizon and intervention values are **scenario assumptions**, not estimates of medical progress. The non-COVID 2021 baseline may retain indirect pandemic effects. Interventions apply additional reductions relative to the chosen baseline; avoid counting the same progress in both the trend and the intervention.

## Reading the results

- **Mean age at death** is initial age plus expected remaining life. It is not life expectancy at birth.
- **p75** is the age by which 75% of the modeled cohort has died. Percentiles are not confidence intervals for the forecast.
- **Unresolved** means that surviving beyond the numerical horizon could materially affect the full mean. The app shows expected remaining years lived *before the horizon*, remaining survival and an upper bound on omitted life-years instead of inventing a constant-risk tail.
- A resolved mean omits at most **0.001 years** under the selected model. This is numerical error control, not forecast accuracy.
- **Beyond horizon** means a percentile was not reached within the calculation; it does not imply immortality.
- Sensitivity rows are deterministic alternatives, not samples from a probability distribution or a confidence band.

## Reference data and reproducibility

The source `mortality.csv` is retained unchanged. The original extraction metadata are not available. New reference data are sourced from Eurostat for Spain, males, 2021:

- [`demo_mlifetable`](https://ec.europa.eu/eurostat/databrowser/view/demo_mlifetable/default/table): Mx (deaths per person-year) and ex (remaining years).
- [`hlth_cd_acdr2`](https://ec.europa.eu/eurostat/databrowser/view/hlth_cd_acdr2/default/table): all-cause deaths per 100,000, matching the source age groups.

`data/reference_metadata.json` records exact requests, retrieval date, source update timestamps, flags, units and SHA-256 checksums. The 95+ Mx is an open-group average, not a single-year observation. Calibration does not explain or remove discrepancies from the source audit.

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

Values are fractions: `0.015` means 1.5% per year. The floor is a multiplier of the cause's **counterfactual rate**, after time trends and the chosen COVID path. Years must be integers in 2021–2120; fast start must not precede slow start; rates must be in [0, 1]; floor in [1e-12, 1]. Invalid and non-finite values are rejected. Active settings can be exported from the simulator.

## Code layout

- `app.py`: entry point, cached inputs and tabs.
- `data.py`: source normalization, validation and reference loading.
- `config.py`: validated, immutable projection assumptions.
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
from lifespan_simulator.trends import build_trend_model

raw = load_mortality_data()
frame = get_simulator_frame(raw)
trend = build_trend_model(raw, frame)
model = MortalityModel(frame, trend.slope_table, load_official_life_table())
result = simulate_survival(model, initial_age=35, config=ProjectionConfig(start_year=2026))
```

Projection functions now accept `MortalityModel` and `ProjectionConfig` instead of a raw DataFrame plus many independent options. `life_expectancy_age` and `remaining_life_expectancy` can be `None` when the full mean cannot be resolved. Use `restricted_remaining_life_expectancy`, `survival_at_end` and `remaining_expectation_bound` to inspect such cases. Export the UI's assumptions JSON alongside sensitivity CSVs to record the chosen settings.
