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
