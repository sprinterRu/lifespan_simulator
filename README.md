# Lifespan Simulator

Interactive Streamlit app for exploring male mortality in Spain using the `mortality.csv` dataset in this folder.

## What the app does

- Simulates a synthetic cohort starting at a chosen age in 2021.
- Lets you apply selected cause-specific interventions with a slow exponential mortality decline followed by a faster exponential decline.
- Lets you dampen historical trend extrapolation so fitted slopes taper and then freeze.
- Shows projected cause-specific mortality paths, total mortality, and a survival curve with death-age percentiles.
- Explores 2011-2021 mortality-rate trends for selected causes and age groups with linear trend lines and 95% confidence intervals.

## Modeling assumptions

- The simulator uses the 2021 mortality schedule only.
- Source five-year age-bucket rates are linearly interpolated across age-bucket midpoints to produce annual attained-age rates; the open-ended `95 years or over` bucket is held flat from age 95 onward.
- Mortality rates are interpreted as constant annual hazards within each modeled one-year age interval.
- With trend dampening enabled, fitted slopes apply fully for 15 years, taper over 15 years, then freeze.
- Interventions multiply the no-intervention cause-specific mortality rate by a two-stage exponential decline: a slow annual reduction, then a fast annual reduction, bounded by an irreducible floor multiplier. The default parameters are slow start 2040, slow rate 0.015, fast start 2080, fast rate 0.10, and floor 0.01.
- The two-stage shape is inspired by historical tuberculosis mortality declines, including the England and Wales series from Our World in Data: https://ourworldindata.org/grapher/death-rate-tuberculosis-england-wales
- Explicit annual projection continues while trends or active interventions are still changing, then the tail hazard is frozen.
- The source file contains overlapping roll-up categories and subcategories. To avoid double-counting in the simulator totals, the app excludes the overlapping roll-ups and keeps a non-overlapping cause set for survival calculations.
- The historical explorer keeps the full dataset because it visualizes individual series rather than summing them.

## Intervention CSV schema

The active interventions CSV uses these columns:

```text
Cause,Slow start year,Slow annual reduction,Fast start year,Fast annual reduction,Floor
```

Annual reductions and floors are fractional values: `0.015` means 1.5% per year, and `0.01` means the intervention cannot reduce that cause below 1% of its no-intervention mortality rate.

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
