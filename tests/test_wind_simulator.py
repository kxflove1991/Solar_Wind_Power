import pytest
import pandas as pd
import numpy as np
from src.wind_simulator import WindSimulator


@pytest.fixture
def sample_weather_df():
    idx = pd.date_range('2020-01-01', periods=8760, freq='h', tz='UTC')
    df = pd.DataFrame({
        'temp_air': np.sin(np.linspace(0, 2 * np.pi * 365, 8760)) * 15 + 10,
        'wind_speed': np.abs(np.sin(np.linspace(0, 2 * np.pi * 365, 8760))) * 12 + 2,
    }, index=idx)
    return df


class TestWindSimulator:
    def test_custom_10mw_simulation(self, sample_weather_df):
        sim = WindSimulator(lat=39.0, lon=102.0, capacity_mw=100.0, hub_height=120)
        power = sim.run(sample_weather_df)

        assert len(power) == len(sample_weather_df)
        assert (power >= 0).all()
        assert power.max() <= 100e6 * 1.01  # 允许微小浮点误差

    def test_integer_turbines(self, sample_weather_df):
        sim = WindSimulator(lat=39.0, lon=102.0, capacity_mw=100.0,
                            hub_height=120, use_integer_turbines=True)
        power = sim.run(sample_weather_df)
        # 100MW / 10MW = 10 台，整数缩放应整好 10 倍
        assert sim.turbine.nominal_power == 10e6
        assert (power >= 0).all()

    def test_missing_wind_speed(self, sample_weather_df):
        sim = WindSimulator(lat=39.0, lon=102.0, capacity_mw=100.0)
        df_missing = sample_weather_df.drop(columns=['wind_speed'])
        with pytest.raises(ValueError, match="wind_speed"):
            sim.run(df_missing)

    def test_missing_temperature(self, sample_weather_df):
        sim = WindSimulator(lat=39.0, lon=102.0, capacity_mw=100.0)
        df_missing = sample_weather_df.drop(columns=['temp_air'])
        with pytest.raises(ValueError, match="temp_air"):
            sim.run(df_missing)

    def test_power_curve_file_loaded(self):
        sim = WindSimulator(lat=39.0, lon=102.0, capacity_mw=100.0, turbine_type='custom_10mw')
        assert sim.turbine.nominal_power == 10e6

    def test_unsupported_turbine_type(self):
        with pytest.raises(ValueError, match="不支持"):
            WindSimulator(lat=39.0, lon=102.0, turbine_type='unknown_type')
