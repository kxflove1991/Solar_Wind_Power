import pytest
import pandas as pd
import numpy as np
from src.analyzer import ResourceAnalyzer


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


class TestResourceAnalyzer:
    def test_analyze_radiation(self, sample_weather_df):
        analyzer = ResourceAnalyzer(sample_weather_df)
        result = analyzer.analyze_radiation()

        assert 'stats' in result
        assert 'sunshine_hours' in result
        assert 'yearly_irradiation' in result
        assert 'avg_temp' in result
        assert result['sunshine_hours'] > 0
        assert result['yearly_irradiation']['ghi'] > 0

    def test_analyze_seasonality(self, sample_weather_df):
        analyzer = ResourceAnalyzer(sample_weather_df)
        result = analyzer.analyze_seasonality()

        assert len(result) == 12
        assert 'ghi' in result.columns
        assert 'dni' in result.columns
        assert 'temp_air' in result.columns

    def test_invalid_input(self):
        df = pd.DataFrame({'a': [1, 2, 3]})
        with pytest.raises(TypeError):
            ResourceAnalyzer(df)

    def test_missing_columns(self):
        df = pd.DataFrame({'ghi': [1, 2, 3]}, index=pd.date_range('2020-01-01', periods=3, freq='h'))
        with pytest.raises(ValueError):
            ResourceAnalyzer(df)
