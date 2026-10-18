import pytest
from streamlit.testing.v1 import AppTest


def test_app_reference_sensitivity_and_scenario_controls():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Simulator",
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
    next(widget for widget in app.selectbox if widget.label == "Mortality after age 94").select(
        "flat"
    ).run()
    assert not app.exception


def test_unresolved_mean_has_no_gain_arrow_and_explains_remaining_survival():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    next(
        widget for widget in app.number_input if widget.label == "Numerical integration end age"
    ).set_value(200)
    causes = next(
        widget for widget in app.multiselect if widget.label == "Causes to modify in the scenario"
    )
    causes.set_value(causes.options).run()
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
