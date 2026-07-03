import pytest
import pandas as pd
import numpy as np
from src.simulator import PVSimulator


@pytest.fixture
def sample_weather_df():
    idx = pd.date_range('2020-01-01', periods=8760, freq='h', tz='UTC')
    df = pd.DataFrame({
        'ghi': np.clip(np.sin(np.linspace(0, 2 * np.pi * 365, 8760)) * 500 + 200, 0, None),
        'dni': np.clip(np.sin(np.linspace(0, 2 * np.pi * 365, 8760)) * 400 + 150, 0, None),
        'dhi': np.clip(np.sin(np.linspace(0, 2 * np.pi * 365, 8760)) * 100 + 50, 0, None),
        'temp_air': np.sin(np.linspace(0, 2 * np.pi * 365, 8760)) * 15 + 10,
        'wind_speed': np.abs(np.sin(np.linspace(0, 2 * np.pi * 365, 8760))) * 10 + 1,
    }, index=idx)
    return df


class TestPVSimulator:
    def test_default_simulation(self, sample_weather_df):
        sim = PVSimulator(lat=39.0, lon=102.0, capacity_mw=100.0)
        results = sim.run(sample_weather_df)

        assert hasattr(results, 'ac')
        assert len(results.ac) == len(sample_weather_df)
        assert (results.ac >= 0).all()
        assert results.ac.max() <= 100e6

    def test_surface_tilt_default_is_latitude(self):
        sim = PVSimulator(lat=25.0, lon=100.0, capacity_mw=10.0)
        assert sim.surface_tilt == 25.0

    def test_invalid_module_name(self):
        with pytest.raises(ValueError, match="不在 SandiaMod"):
            PVSimulator(lat=39.0, lon=102.0, module_name='NonExistentModuleXYZ')

    def test_invalid_inverter_name(self):
        with pytest.raises(ValueError, match="不在 CEC"):
            PVSimulator(lat=39.0, lon=102.0, inverter_name='NonExistentInverterXYZ')
