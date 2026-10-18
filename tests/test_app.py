from math import exp

import pytest
from streamlit.testing.v1 import AppTest

from lifespan_simulator.data import DEMENTIA_CODE, DEMENTIA_NAME


def test_app_reference_sensitivity_and_scenario_controls():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Simulator",
        "Cause trajectories",
        "Reference checks",
        "Historical explorer",
    ]
    next(
        widget for widget in app.multiselect if widget.label == "Causes to modify in the scenario"
    ).select("Malignant neoplasm of trachea, bronchus and lung").run()
    assert not app.exception
    next(
        widget for widget in app.checkbox if widget.label == "Run sensitivity analysis"
    ).check().run()
    assert not app.exception
    assert any("Assumption" in table.value.columns for table in app.dataframe)
    next(widget for widget in app.slider if widget.label == "Initial age").set_value(95).run()
    assert not app.exception
    assert any("Ranking uses extrapolated rates" in item.value for item in app.caption)
    next(widget for widget in app.selectbox if widget.label == "Mortality after age 94").select(
        "flat"
    ).run()
    assert not app.exception


def test_unresolved_mean_has_no_gain_arrow_and_explains_remaining_survival():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    next(widget for widget in app.selectbox if widget.label == "Mortality after age 94").select(
        "kannisto"
    )
    next(
        widget for widget in app.number_input if widget.label == "Numerical integration end age"
    ).set_value(200)
    causes = next(
        widget for widget in app.multiselect if widget.label == "Causes to modify in the scenario"
    )
    causes.set_value([name for name in causes.options if name != "Falls"]).run()
    assert not app.exception

    mean = next(metric for metric in app.metric if metric.label == "Scenario mean age at death")
    median = next(metric for metric in app.metric if metric.label == "Scenario median age")
    assert mean.value == "Unresolved"
    assert mean.delta == ""
    assert float(median.value) > 35
    assert any("Simulation completed" in warning.value for warning in app.warning)
    assert any("median can still be known" in warning.value for warning in app.warning)

    diagnostics = next(
        table.value for table in app.dataframe if "Full mean resolved" in table.value.columns
    ).set_index("Projection")
    scenario = diagnostics.loc["Scenario"]
    lower = scenario["Mean age at death: lower bound"]
    upper = scenario["Mean age at death: upper bound"]
    assert lower == pytest.approx(35 + scenario["Expected remaining years lived before 200"])
    assert upper - lower == pytest.approx(scenario["Unintegrated life-years upper bound"])
    assert any(
        f"Scenario: {scenario['Alive at 200 (%)']:.2f}%" in caption.value for caption in app.caption
    )
    next(
        widget for widget in app.number_input if widget.label == "Numerical integration end age"
    ).set_value(2_000).run()
    assert not app.exception
    mean = next(metric for metric in app.metric if metric.label == "Scenario mean age at death")
    assert lower - 0.05 <= float(mean.value) <= upper + 0.05
    assert float(mean.delta) > 0


def test_floor_free_fast_displays_infinite_mean_separately_from_horizon_failure():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    next(widget for widget in app.selectbox if widget.label == "Mortality after age 94").select(
        "kannisto"
    )
    causes = next(
        widget for widget in app.multiselect if widget.label == "Causes to modify in the scenario"
    )
    causes.set_value(causes.options).run()
    assert not app.exception
    mean = next(metric for metric in app.metric if metric.label == "Scenario mean age at death")
    assert mean.value == "∞" and mean.delta == ""
    assert any("full mean is infinite under this model" in item.value for item in app.info)
    assert not any("not resolved" in item.value for item in app.warning)
    assert any(widget.label.startswith("Slow-stage floor") for widget in app.number_input)
    next(
        widget for widget in app.checkbox if widget.label == "Run sensitivity analysis"
    ).check().run()
    assert not app.exception
    sensitivity = next(
        table.value for table in app.dataframe if "Assumption" in table.value.columns
    )
    assert sensitivity.iloc[0]["Mean status"] == "Infinite"


def test_individual_tail_overrides_are_visible_and_reset_to_automatic_fit():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    causes = next(w for w in app.multiselect if w.label == "Causes to modify in the scenario")
    assert DEMENTIA_NAME in causes.options
    assert "Dementia" not in causes.options and "Alzheimer disease" not in causes.options
    custom = next(w for w in app.multiselect if w.label == "Causes with custom age trajectories")
    custom.select(DEMENTIA_NAME).run()
    next(w for w in app.selectbox if w.label == f"Tail family for {DEMENTIA_NAME}").select(
        "exponent"
    )
    next(w for w in app.number_input if w.label == f"Tail slope for {DEMENTIA_NAME}").set_value(
        0.15
    )
    next(w for w in app.slider if w.label == "Declining-tail residual share").set_value(0.25).run()
    assert not app.exception
    table = next(t.value for t in app.dataframe if "Family" in t.value.columns).set_index("Cause")
    assert table.loc[DEMENTIA_NAME, "Family"] == "exponent"
    assert table.loc[DEMENTIA_NAME, "Status"] == "Manual override"
    assert table.loc[DEMENTIA_NAME, "Slope (1/year)"] == pytest.approx(0.15)
    assert any("selected residual share: 25%" in c.value for c in app.caption)
    next(w for w in app.selectbox if w.label == "Inspect a cause trajectory").select(
        "Malignant neoplasm of trachea, bronchus and lung"
    ).run()
    assert not app.exception
    next(w for w in app.multiselect if w.label == "Causes with custom age trajectories").set_value(
        []
    ).run()
    assert not app.exception
    table = next(t.value for t in app.dataframe if "Family" in t.value.columns).set_index("Cause")
    assert table.loc[DEMENTIA_NAME, "Status"] != "Manual override"


def test_dementia_comparison_switch_uses_saved_sigmoid_and_restores_default():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    before = next(t.value for t in app.dataframe if "Family" in t.value.columns).set_index("Cause")
    assert before.loc[DEMENTIA_NAME, "Family"] == "exponent"
    assert before.loc[DEMENTIA_NAME, "Age window"] == "70–89"
    next(w for w in app.radio if w.label == "Dementia age trajectory").set_value(
        "Sigmoid fitted at 70–94 (comparison)"
    ).run()
    assert not app.exception
    comparison = next(t.value for t in app.dataframe if "Family" in t.value.columns).set_index(
        "Cause"
    )
    row = comparison.loc[DEMENTIA_NAME]
    assert row.Family == "sigmoid" and row.Status == "Comparison scenario"
    assert row["Slope (1/year)"] == pytest.approx(0.2167405741)
    assert row["Sigmoid plateau / age-94 rate"] == pytest.approx(1.528725466)
    assert row["Alternative"] == "exponent"
    assert (before.drop(DEMENTIA_NAME).Family == comparison.drop(DEMENTIA_NAME).Family).all()
    next(w for w in app.radio if w.label == "Dementia age trajectory").set_value(
        "Exponent fitted at 70–89 (default)"
    ).run()
    assert not app.exception
    restored = next(t.value for t in app.dataframe if "Family" in t.value.columns).set_index(
        "Cause"
    )
    assert restored.loc[DEMENTIA_NAME, "Family"] == "exponent"
    assert (
        restored.loc[DEMENTIA_NAME, "Slope (1/year)"] == before.loc[DEMENTIA_NAME, "Slope (1/year)"]
    )


def test_select_exponential_causes_adds_active_exponents_and_keeps_previous_selection():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    lung = "Malignant neoplasm of trachea, bronchus and lung"
    next(w for w in app.multiselect if w.label == "Causes to modify in the scenario").select(
        lung
    ).run()
    table = next(t.value for t in app.dataframe if "Family" in t.value.columns)
    expected = set(table.loc[(table.Family == "exponent") & (table["Slope (1/year)"] > 0), "Cause"])
    next(
        w for w in app.button if w.label == "Select all exponentially growing causes"
    ).click().run()
    assert not app.exception
    selected = next(
        w for w in app.multiselect if w.label == "Causes to modify in the scenario"
    ).value
    assert set(selected) == expected | {lung}
    assert DEMENTIA_NAME in selected
    dementia_rate = next(
        w for w in app.number_input if w.label == f"Fast annual reduction for {DEMENTIA_NAME}"
    ).value
    assert dementia_rate == pytest.approx(1 - 0.9 * exp(-0.1980141115))
    assert next(
        w for w in app.number_input if w.label == f"Fast annual reduction for {lung}"
    ).value == pytest.approx(0.1)


def test_fast_defaults_follow_shapes_but_manual_and_imported_values_are_preserved():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    next(w for w in app.multiselect if w.label == "Causes to modify in the scenario").select(
        DEMENTIA_NAME
    ).run()
    next(w for w in app.radio if w.label == "Dementia age trajectory").set_value(
        "Sigmoid fitted at 70–94 (comparison)"
    ).run()
    rate_label = f"Fast annual reduction for {DEMENTIA_NAME}"
    assert next(w for w in app.number_input if w.label == rate_label).value == pytest.approx(0.1)
    next(w for w in app.number_input if w.label == rate_label).set_value(0.4).run()
    next(w for w in app.radio if w.label == "Dementia age trajectory").set_value(
        "Exponent fitted at 70–89 (default)"
    ).run()
    assert not app.exception
    assert next(w for w in app.number_input if w.label == rate_label).value == pytest.approx(0.4)
    next(w for w in app.button if w.label == "Reset selected fast rates to defaults").click().run()
    assert next(w for w in app.number_input if w.label == rate_label).value == pytest.approx(
        1 - 0.9 * exp(-0.1980141115)
    )

    # Exercise the actual CSV-to-widget seeding function before rendering.
    imported = AppTest.from_string(
        """
from app import main
from lifespan_simulator.ui import apply_loaded_interventions
from lifespan_simulator.data import DEMENTIA_CODE, DEMENTIA_NAME
import streamlit as st
if "seeded" not in st.session_state:
    apply_loaded_interventions([DEMENTIA_NAME], {DEMENTIA_CODE: {
        "slow_start_year": 2040, "slow_rate": .015, "fast_start_year": 2080,
        "fast_rate": .12, "floor": .01,
    }}, {DEMENTIA_NAME: DEMENTIA_CODE})
    st.session_state.seeded = True
main()
""",
        default_timeout=30,
    ).run()
    assert not imported.exception
    assert imported.session_state[f"fast_rate_custom_{DEMENTIA_CODE}"]
    assert next(w for w in imported.number_input if w.label == rate_label).value == pytest.approx(
        0.12
    )
