# Lifespan Simulator

Interactive Streamlit app for exploring male mortality in Spain using the `mortality.csv` dataset in this folder.

## What the app does

- Simulates a synthetic cohort starting at a chosen age in 2021.
- Lets you override selected cause-specific death rates from a future calendar year onward.
- Lets you dampen historical trend extrapolation so fitted slopes taper and then freeze.
- Shows projected cause-specific mortality paths, total mortality, and a survival curve with death-age percentiles.
- Explores 2011-2021 mortality-rate trends for selected causes and age groups with linear trend lines and 95% confidence intervals.

## Modeling assumptions

- The simulator uses the 2021 mortality schedule only.
- Mortality rates are interpreted as constant annual hazards within each age bucket.
- With trend dampening enabled, fitted slopes apply fully for 15 years, taper over 15 years, then freeze.
- The `95 years or over` bucket remains flat after age 94.
- The source file contains overlapping roll-up categories and subcategories. To avoid double-counting in the simulator totals, the app excludes the overlapping roll-ups and keeps a non-overlapping cause set for survival calculations.
- The historical explorer keeps the full dataset because it visualizes individual series rather than summing them.

## Run

1. Install `uv` if it is not already available.
2. Sync the project environment:

```bash
uv sync
```

3. Launch the app:

```bash
uv run streamlit run app.py
```

Or use the helper script:

```bash
./run_app.sh
```

## File layout

- `app.py`: Streamlit entry point.
- `lifespan_simulator/data.py`: Dataset loading and normalization.
- `lifespan_simulator/simulation.py`: Cohort survival and mortality projection logic.
- `lifespan_simulator/trends.py`: Linear trend and confidence-interval calculations.
- `lifespan_simulator/charts.py`: Plotly chart builders.
