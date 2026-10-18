from io import StringIO

import pytest

from lifespan_simulator.data import DATA_PATH, load_mortality_data
from lifespan_simulator.interventions import build_scenario_map
from lifespan_simulator.scenario_io import active_interventions_frame, parse_interventions_csv


def test_intervention_csv_round_trip():
    original = build_scenario_map({"C33_C34": (2040, 0.015, 2080, 0.1, 0.01)})
    csv = active_interventions_frame(original, {"C33_C34": "Lung cancer"}).to_csv(index=False)
    names, values = parse_interventions_csv(StringIO(csv), {"Lung cancer": "C33_C34"})
    assert names == ["Lung cancer"]
    assert build_scenario_map(values) == original


@pytest.mark.parametrize(
    "column,value",
    [
        ("Slow start year", "inf"),
        ("Fast start year", "nan"),
        ("Slow annual reduction", "-0.1"),
        ("Floor", "0"),
        ("Floor", "1e-30"),
        ("Fast annual reduction", "inf"),
    ],
)
def test_bad_csv_values_raise_user_visible_errors(column, value):
    scenarios = build_scenario_map({"X": (2040, 0.015, 2080, 0.1, 0.01)})
    table = active_interventions_frame(scenarios, {"X": "Example"}).astype(str)
    table.loc[0, column] = value
    with pytest.raises(ValueError):
        parse_interventions_csv(StringIO(table.to_csv(index=False)), {"Example": "X"})


def test_source_duplicates_and_missing_rates_are_rejected(tmp_path):
    lines = DATA_PATH.read_text().splitlines()
    path = tmp_path / "bad.csv"
    path.write_text("\n".join([lines[0], lines[1], lines[1]]) + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        load_mortality_data(path)
    path.write_text(lines[0] + "\n" + lines[1].rsplit(",", 1)[0] + ",\n")
    with pytest.raises(ValueError, match="Missing"):
        load_mortality_data(path)


def test_identical_legacy_dementia_rows_merge_losslessly():
    from lifespan_simulator.data import DEMENTIA_CODE, DEMENTIA_NAME

    original = build_scenario_map({code: (2040, 0.015, 2080, 0.1, 0.01) for code in ["D", "A"]})
    table = active_interventions_frame(original, {"D": "Dementia", "A": "Alzheimer disease"})
    names, values = parse_interventions_csv(
        StringIO(table.to_csv(index=False)), {DEMENTIA_NAME: DEMENTIA_CODE}
    )
    assert names == [DEMENTIA_NAME]
    assert build_scenario_map(values)[DEMENTIA_CODE] == original["D"]


def test_single_legacy_dementia_intervention_is_not_silently_broadened():
    from lifespan_simulator.data import DEMENTIA_CODE, DEMENTIA_NAME

    original = build_scenario_map({"D": (2040, 0.015, 2080, 0.1, 0.01)})
    table = active_interventions_frame(original, {"D": "Dementia"})
    with pytest.raises(ValueError, match="both causes have identical"):
        parse_interventions_csv(StringIO(table.to_csv(index=False)), {DEMENTIA_NAME: DEMENTIA_CODE})


def test_conflicting_legacy_dementia_rows_require_explicit_combined_settings():
    from lifespan_simulator.data import DEMENTIA_CODE, DEMENTIA_NAME

    original = build_scenario_map(
        {
            "D": (2040, 0.015, 2080, 0.1, 0.01),
            "A": (2040, 0.015, 2080, 0.2, 0.01),
        }
    )
    table = active_interventions_frame(original, {"D": "Dementia", "A": "Alzheimer disease"})
    with pytest.raises(ValueError, match="both causes have identical"):
        parse_interventions_csv(StringIO(table.to_csv(index=False)), {DEMENTIA_NAME: DEMENTIA_CODE})
