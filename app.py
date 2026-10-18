"""Streamlit entry point for the lifespan scenario explorer."""

import streamlit as st

from lifespan_simulator.data import (
    DATA_PATH,
    REFERENCE_DIR,
    get_simulator_frame,
    load_mortality_data,
    load_official_life_table,
)
from lifespan_simulator.tails import fit_cause_tails
from lifespan_simulator.trends import build_trend_model
from lifespan_simulator.ui import (
    apply_custom_styles,
    render_explorer_tab,
    render_simulator_tab,
    render_trajectories_tab,
    render_validation_tab,
)


@st.cache_data(show_spinner=False)
def load_inputs(data_versions: tuple[int, ...]):
    """Cache preprocessing; file modification times invalidate the snapshot."""
    raw = load_mortality_data()
    simulator = get_simulator_frame(raw)
    official = load_official_life_table()
    tails = fit_cause_tails(raw, tuple(sorted(simulator.cause_code.unique())))
    return raw, simulator, official, build_trend_model(raw, simulator, end_year=2019), tails


def main() -> None:
    st.set_page_config(page_title="Lifespan Simulator", layout="wide")
    apply_custom_styles()
    st.title("Lifespan Simulator")
    st.caption(
        "Male mortality in Spain. Explore conditional lifespan scenarios, compare official reference tables, "
        "and test how conclusions change with mortality assumptions."
    )
    paths = (
        DATA_PATH,
        REFERENCE_DIR / "spain_male_2021_life_table.csv",
        REFERENCE_DIR / "spain_male_2021_all_causes.csv",
        *(
            DATA_PATH.parent / "lifespan_simulator" / file
            for file in ("data.py", "tails.py", "trends.py")
        ),
    )
    raw, simulator, official, trends, tails = load_inputs(
        tuple(path.stat().st_mtime_ns for path in paths)
    )
    simulator_tab, trajectories_tab, validation_tab, explorer_tab = st.tabs(
        ["Simulator", "Cause trajectories", "Reference checks", "Historical explorer"]
    )
    with simulator_tab:
        active_tails, config = render_simulator_tab(simulator, official, trends, tails)
    with trajectories_tab:
        render_trajectories_tab(raw, simulator, official, active_tails, config)
    with validation_tab:
        render_validation_tab(simulator, official, active_tails, config)
    with explorer_tab:
        render_explorer_tab(raw)


if __name__ == "__main__":
    main()
