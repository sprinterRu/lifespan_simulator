"""Load and normalize the mortality dataset."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "mortality.csv"
REFERENCE_DIR = DATA_PATH.parent / "data"
DEMENTIA_CODE = "F01_F03_G30"
DEMENTIA_NAME = "Dementia including Alzheimer disease"
DEMENTIA_COMPONENTS = frozenset({"F01_F03", "G30"})

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

    required = {"age_code", "age_label", "cause_code", "cause_name", "year", "rate"}
    if not required.issubset(frame):
        raise ValueError(f"Missing mortality columns: {sorted(required - set(frame))}")
    if frame[list(required)].isna().any().any():
        raise ValueError("Missing mortality observation; absence is not a zero rate")
    if frame.duplicated(["year", "age_code", "cause_code"]).any():
        raise ValueError("Duplicate year/age/cause observations")

    age_parts = frame.apply(
        lambda row: _parse_age_bucket(row["age_code"], row["age_label"]),
        axis=1,
        result_type="expand",
    )
    age_parts.columns = ["age_start", "age_end", "age_display"]

    frame = pd.concat([frame, age_parts], axis=1)
    frame["year"] = frame["year"].astype(int)
    frame["rate"] = frame["rate"].astype(float)
    if not np.isfinite(frame["rate"]).all() or (frame["rate"] < 0).any():
        raise ValueError("Mortality rates must be finite and non-negative")
    frame["age_mid"] = frame["age_start"] + frame["age_end"].fillna(99).sub(frame["age_start"]).div(
        2
    )

    return frame.sort_values(["year", "age_start", "cause_name"]).reset_index(drop=True)


def get_simulator_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the 2021 non-overlapping cause set used for survival simulation."""
    simulator = frame[
        (frame["year"] == 2021)
        & (frame.age_start < 95)
        & (~frame["cause_code"].isin(SIMULATOR_EXCLUDED_CAUSES))
    ].copy()
    expected_ages = set(range(35, 95, 5))
    if simulator.empty or any(
        set(group.age_start) != expected_ages for _, group in simulator.groupby("cause_code")
    ):
        raise ValueError("Every 2021 simulator cause must cover closed groups 35–94")
    return merge_dementia(simulator).sort_values(["age_start", "cause_name"]).reset_index(drop=True)


def merge_dementia(frame: pd.DataFrame) -> pd.DataFrame:
    """Sum disjoint underlying-cause codes before fitting any age or time trend.

    The original history remains available in the explorer. Reject incomplete
    pairs instead of interpreting an absent Alzheimer's/dementia cell as zero.
    """
    members = frame[frame.cause_code.isin(DEMENTIA_COMPONENTS)]
    if members.empty:
        return frame.copy()
    if DEMENTIA_CODE in set(frame.cause_code):
        raise ValueError("Combined dementia and its components must not coexist")
    keys = ["year", "age_start"]
    if not (members.groupby(keys).cause_code.nunique() == 2).all():
        raise ValueError("Dementia merge requires both component causes at every year/age")
    combined = members[members.cause_code == "F01_F03"].copy().set_index(keys)
    combined["rate"] = members.groupby(keys).rate.sum()
    combined["cause_code"] = DEMENTIA_CODE
    combined["cause_name"] = DEMENTIA_NAME
    return pd.concat(
        [frame[~frame.cause_code.isin(DEMENTIA_COMPONENTS)], combined.reset_index()],
        ignore_index=True,
    )


def get_age_bucket_start(age: int) -> int:
    """Map an attained age to the dataset's age bucket."""
    if age >= 90:
        return 90
    return 35 + ((age - 35) // 5) * 5


def get_display_horizon(initial_age: int) -> int:
    """Pick a chart horizon that is readable for both younger and older cohorts."""
    return min(max(110, initial_age + 20), 125)


def load_official_life_table() -> pd.DataFrame:
    """Read the checked-in Eurostat 2021 Mx/ex snapshot without network access."""
    frame = pd.read_csv(REFERENCE_DIR / "spain_male_2021_life_table.csv", dtype={"age": str})
    return frame[frame.age != "95+"].reset_index(drop=True)


def load_official_cause_totals() -> pd.DataFrame:
    frame = pd.read_csv(REFERENCE_DIR / "spain_male_2021_all_causes.csv")
    return frame[frame.age_code != "Y_GE95"].reset_index(drop=True)
