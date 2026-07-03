import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from src.visualizer import Visualizer


@pytest.fixture
def sample_output_df():
    idx = pd.date_range('2020-01-01', periods=8760, freq='h', tz='Asia/Shanghai')
    df = pd.DataFrame({
        'PU': np.clip(np.sin(np.linspace(0, 2 * np.pi * 365, 8760)), 0, 1),
        'Wind_PU': np.abs(np.sin(np.linspace(0, 2 * np.pi * 365, 8760))),
    }, index=idx)
    return df


@pytest.fixture
def sample_weather_df():
    idx = pd.date_range('2020-01-01', periods=8760, freq='h', tz='Asia/Shanghai')
    df = pd.DataFrame({
        'ghi': np.clip(np.sin(np.linspace(0, 2 * np.pi * 365, 8760)) * 500 + 200, 0, None),
        'dni': np.clip(np.sin(np.linspace(0, 2 * np.pi * 365, 8760)) * 400 + 150, 0, None),
        'dhi': np.clip(np.sin(np.linspace(0, 2 * np.pi * 365, 8760)) * 100 + 50, 0, None),
        'temp_air': np.sin(np.linspace(0, 2 * np.pi * 365, 8760)) * 15 + 10,
        'wind_speed': np.abs(np.sin(np.linspace(0, 2 * np.pi * 365, 8760))) * 10 + 1,
    }, index=idx)
    return df


class TestVisualizer:
    def test_prepare_heatmap_data_shape(self, sample_output_df):
        matrix = Visualizer._prepare_heatmap_data(sample_output_df['PU'])
        assert matrix.shape == (24, 365)

    def test_pv_heatmap(self, sample_output_df, tmp_path):
        Visualizer.plot_pv_heatmap(sample_output_df, tmp_path)
        assert (tmp_path / "pv_heatmap.png").exists()

    def test_pv_duration_curve(self, sample_output_df, tmp_path):
        Visualizer.plot_pv_duration_curve(sample_output_df, tmp_path)
        assert (tmp_path / "pv_duration_curve.png").exists()

    def test_wind_power_analysis(self, sample_output_df, tmp_path):
        Visualizer.plot_wind_power_analysis(sample_output_df, tmp_path)
        assert (tmp_path / "wind_heatmap.png").exists()
        assert (tmp_path / "wind_duration_curve.png").exists()

    def test_weather_trends(self, sample_weather_df, tmp_path):
        Visualizer.plot_weather_trends(sample_weather_df, tmp_path)
        assert (tmp_path / "weather_trends_yearly.png").exists()

    def test_weather_histograms(self, sample_weather_df, tmp_path):
        Visualizer.plot_weather_histograms(sample_weather_df, tmp_path)
        assert (tmp_path / "weather_histograms.png").exists()

    def test_wind_analysis(self, sample_weather_df, tmp_path):
        Visualizer.plot_wind_analysis(sample_weather_df, tmp_path)
        assert (tmp_path / "wind_speed_distribution.png").exists()
        assert (tmp_path / "wind_speed_profiles.png").exists()

    def test_radiation_heatmaps(self, sample_weather_df, tmp_path):
        Visualizer.plot_radiation_heatmaps(sample_weather_df, tmp_path)
        assert (tmp_path / "radiation_heatmaps.png").exists()
