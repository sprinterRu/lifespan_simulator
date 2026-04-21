"""Load and normalize the mortality dataset."""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "mortality.csv"

# The source mixes mutually exclusive causes with overlapping roll-ups.
# These roll-ups are excluded from the simulator totals to avoid double-counting.
SIMULATOR_EXCLUDED_CAUSES = {
    "ACC": "Aggregate accidents bucket overlaps with transport, falls, drowning, poisoning and accident remainder series.",
    "B180-B182": "Subset of viral hepatitis; parent viral hepatitis series is retained instead.",
    "G_H": "Aggregate nervous-system bucket overlaps with Parkinson, Alzheimer and remainder series.",
    "I20-I25": "Aggregate ischaemic-heart-disease bucket overlaps with infarction and other ischaemic-heart-disease subseries.",
    "J40-J47": "Aggregate chronic-lower-respiratory bucket overlaps with asthma and other chronic lower respiratory subseries.",
    "K72-K75": "Partially overlaps with the broader chronic liver disease series retained for totals.",
    "M": "Aggregate musculoskeletal bucket overlaps with arthrosis/rheumatoid arthritis and remainder series.",
    "R": "Aggregate symptoms/findings bucket overlaps with ill-defined causes and remainder series.",
}

AGE_PATTERN = re.compile(r"Y(?P<start>\d+)-(?P<end>\d+)")


def _parse_age_bucket(age_code: str, age_label: str) -> tuple[int, int | None, str]:
    """Convert source age buckets into numeric ranges and compact labels."""
    if age_code == "Y_GE95":
        return 95, None, "95+"

    match = AGE_PATTERN.fullmatch(age_code)
    if not match:
        raise ValueError(f"Unsupported age bucket: {age_code} ({age_label})")

    start = int(match.group("start"))
    end = int(match.group("end"))
    return start, end, f"{start}-{end}"


def load_mortality_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the mortality CSV and add normalized columns used by the app."""
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame = frame.rename(
        columns={
            "age": "age_code",
            "Age class": "age_label",
            "icd10": "cause_code",
            "International Statistical Classification of Diseases and Related Health Problems (ICD-10)": "cause_name",
            "TIME_PERIOD": "year",
            "OBS_VALUE": "rate",
        }
    )

    age_parts = frame.apply(
        lambda row: _parse_age_bucket(row["age_code"], row["age_label"]), axis=1, result_type="expand"
    )
    age_parts.columns = ["age_start", "age_end", "age_display"]

    frame = pd.concat([frame, age_parts], axis=1)
    frame["year"] = frame["year"].astype(int)
    frame["rate"] = frame["rate"].astype(float)
    frame["age_mid"] = frame["age_start"] + frame["age_end"].fillna(99).sub(frame["age_start"]).div(2)

    return frame.sort_values(["year", "age_start", "cause_name"]).reset_index(drop=True)


def get_simulator_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the 2021 non-overlapping cause set used for survival simulation."""
    simulator = frame[(frame["year"] == 2021) & (~frame["cause_code"].isin(SIMULATOR_EXCLUDED_CAUSES))].copy()
    return simulator.sort_values(["age_start", "cause_name"]).reset_index(drop=True)


def get_age_bucket_start(age: int) -> int:
    """Map an attained age to the dataset's age bucket."""
    if age >= 95:
        return 95
    return 35 + ((age - 35) // 5) * 5


def get_display_horizon(initial_age: int) -> int:
    """Pick a chart horizon that is readable for both younger and older cohorts."""
    return min(max(110, initial_age + 20), 125)

