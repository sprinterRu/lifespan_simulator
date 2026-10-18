import pandas as pd
import pytest

from lifespan_simulator.config import PandemicConfig, ProjectionConfig
from lifespan_simulator.data import (
    get_simulator_frame,
    load_mortality_data,
    load_official_life_table,
)
from lifespan_simulator.mortality import MortalityModel
from lifespan_simulator.trends import build_trend_model


@pytest.fixture(scope="session")
def inputs():
    raw = load_mortality_data()
    frame = get_simulator_frame(raw)
    official = load_official_life_table()
    return raw, frame, official


@pytest.fixture(scope="session")
def model(inputs):
    raw, frame, official = inputs
    return MortalityModel(frame, build_trend_model(raw, frame).slope_table, official)


@pytest.fixture
def constant_model():
    frame = pd.DataFrame(
        [
            {
                "cause_code": "X",
                "cause_name": "Constant risk",
                "age_start": age,
                "age_mid": age + 2,
                "rate": 10_000.0,
            }
            for age in range(35, 100, 5)
        ]
    )
    return MortalityModel(frame)


@pytest.fixture
def fixed_config():
    return ProjectionConfig(
        calibrate=False, old_age_model="flat", pandemic=PandemicConfig(mode="hold_2021")
    )
