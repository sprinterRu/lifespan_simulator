"""CSV interchange for interventions; independent of Streamlit."""

from __future__ import annotations

from math import isfinite

import pandas as pd

from .data import DEMENTIA_NAME
from .interventions import ScenarioMap

ACTIVE_INTERVENTION_COLUMNS = (
    "Cause",
    "Slow start year",
    "Slow annual reduction",
    "Fast start year",
    "Fast annual reduction",
    "Floor",
)


def active_interventions_frame(
    scenario_map: ScenarioMap,
    cause_name_lookup: dict[str, str],
) -> pd.DataFrame:
    """Convert active scenario settings into the CSV-compatible table shape."""
    return pd.DataFrame(
        [
            {
                "Cause": cause_name_lookup[cause_code],
                "Slow start year": config.slow_start_year,
                "Slow annual reduction": config.slow_rate,
                "Fast start year": config.fast_start_year,
                "Fast annual reduction": config.fast_rate,
                "Floor": config.floor,
            }
            for cause_code, config in scenario_map.items()
        ],
        columns=ACTIVE_INTERVENTION_COLUMNS,
    )


def parse_interventions_csv(
    uploaded_file,
    all_cause_options: dict[str, str],
) -> tuple[list[str], dict[str, dict[str, float | int]]]:
    """Read and validate an uploaded Active interventions CSV file."""
    uploaded_file.seek(0)
    frame = pd.read_csv(uploaded_file)
    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame.dropna(how="all")

    missing_columns = [
        column for column in ACTIVE_INTERVENTION_COLUMNS if column not in frame.columns
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required column(s): {missing}.")

    if frame.empty:
        raise ValueError("The uploaded CSV does not contain any interventions.")

    frame = _merge_legacy_dementia_rows(frame, all_cause_options)

    cause_names: list[str] = []
    loaded_inputs: dict[str, dict[str, float | int]] = {}

    for row_number, row in enumerate(frame.to_dict(orient="records"), start=2):
        cause_name = str(row["Cause"]).strip()
        if cause_name not in all_cause_options:
            raise ValueError(f"Unknown cause on row {row_number}: {cause_name}.")
        if cause_name in cause_names:
            raise ValueError(f"Duplicate cause on row {row_number}: {cause_name}.")

        slow_start_year = parse_csv_year(row, "Slow start year", row_number)
        fast_start_year = parse_csv_year(row, "Fast start year", row_number)
        if fast_start_year < slow_start_year:
            raise ValueError(
                f"Fast start year must be no earlier than slow start year on row {row_number}."
            )

        slow_rate = parse_csv_fraction(row, "Slow annual reduction", row_number)
        fast_rate = parse_csv_fraction(row, "Fast annual reduction", row_number)
        floor = parse_csv_fraction(row, "Floor", row_number, allow_zero=False)

        cause_names.append(cause_name)
        loaded_inputs[all_cause_options[cause_name]] = {
            "slow_start_year": slow_start_year,
            "slow_rate": slow_rate,
            "fast_start_year": fast_start_year,
            "fast_rate": fast_rate,
            "floor": floor,
        }

    return cause_names, loaded_inputs


def _merge_legacy_dementia_rows(frame: pd.DataFrame, options: dict[str, str]) -> pd.DataFrame:
    """Migrate only lossless pairs; never silently broaden a one-cause treatment."""
    if DEMENTIA_NAME not in options:
        return frame
    names = frame.Cause.astype(str).str.strip()
    legacy = names.isin(["Dementia", "Alzheimer disease"])
    if not legacy.any():
        return frame
    rows = frame.loc[legacy]
    matching = (
        len(rows) == 2
        and set(names[legacy]) == {"Dementia", "Alzheimer disease"}
        and not names.eq(DEMENTIA_NAME).any()
        and all(
            pd.to_numeric(rows[column], errors="coerce").nunique(dropna=False) == 1
            for column in ACTIVE_INTERVENTION_COLUMNS[1:]
        )
    )
    if not matching:
        raise ValueError(
            f"Dementia and Alzheimer disease are now combined. Use one '{DEMENTIA_NAME}' row. "
            "Legacy rows can be merged only when both causes have identical settings."
        )
    combined = rows.iloc[[0]].copy()
    combined["Cause"] = DEMENTIA_NAME
    return pd.concat([frame.loc[~legacy], combined], ignore_index=True)


def parse_csv_year(row: dict[str, object], column: str, row_number: int) -> int:
    """Parse and validate one intervention year from an uploaded CSV row."""
    value = pd.to_numeric(row[column], errors="coerce")
    if pd.isna(value) or not isfinite(float(value)) or int(value) != value:
        raise ValueError(f"{column} must be a whole number on row {row_number}.")
    parsed = int(value)
    if not 2021 <= parsed <= 2120:
        raise ValueError(f"{column} must be between 2021 and 2120 on row {row_number}.")
    return parsed


def parse_csv_fraction(
    row: dict[str, object],
    column: str,
    row_number: int,
    allow_zero: bool = True,
) -> float:
    """Parse a fractional rate or floor from an uploaded CSV row."""
    value = pd.to_numeric(row[column], errors="coerce")
    range_text = "between 0 and 1" if allow_zero else "between 1e-12 and 1"
    if pd.isna(value) or not isfinite(float(value)):
        raise ValueError(f"{column} must be a number {range_text} on row {row_number}.")

    parsed = float(value)
    lower_bound_valid = parsed >= 0 if allow_zero else parsed >= 1e-12
    if not lower_bound_valid or parsed > 1:
        raise ValueError(
            f"{column} must be a fraction {range_text} on row {row_number} "
            "(for example, 0.015 for 1.5%)."
        )
    return parsed
