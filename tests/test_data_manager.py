import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.data_manager import DataManager, REQUIRED_COLUMNS


@pytest.fixture
def sample_weather_df():
    """生成非闰年 8760 小时气象数据"""
    idx = pd.date_range('2020-01-01', periods=8760, freq='h', tz='UTC')
    df = pd.DataFrame({
        'ghi': np.clip(np.sin(np.linspace(0, 2 * np.pi * 365, 8760)) * 500 + 200, 0, None),
        'dni': np.clip(np.sin(np.linspace(0, 2 * np.pi * 365, 8760)) * 400 + 150, 0, None),
        'dhi': np.clip(np.sin(np.linspace(0, 2 * np.pi * 365, 8760)) * 100 + 50, 0, None),
        'temp_air': np.sin(np.linspace(0, 2 * np.pi * 365, 8760)) * 15 + 10,
        'wind_speed': np.abs(np.sin(np.linspace(0, 2 * np.pi * 365, 8760))) * 10 + 1,
    }, index=idx)
    return df


class TestDataManager:
    def test_load_from_cache(self, tmp_path, sample_weather_df):
        cache_dir = tmp_path / 'cache'
        raw_dir = tmp_path / 'raw'
        dm = DataManager(cache_dir=str(cache_dir), raw_dir=str(raw_dir))

        # 预写缓存
        dm.save_raw_data(sample_weather_df, 39.0, 102.0, 2020)

        df = dm.get_weather_data(39.0, 102.0, 2020, use_cache=True)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert all(col in df.columns for col in REQUIRED_COLUMNS)
        assert len(df) == 8760

    def test_leap_day_removed(self, tmp_path):
        cache_dir = tmp_path / 'cache'
        raw_dir = tmp_path / 'raw'
        dm = DataManager(cache_dir=str(cache_dir), raw_dir=str(raw_dir))

        # 构造闰年 8784 小时数据
        idx = pd.date_range('2020-01-01', periods=8784, freq='h', tz='UTC')
        df = pd.DataFrame({
            'ghi': np.random.rand(8784) * 500,
            'dni': np.random.rand(8784) * 400,
            'dhi': np.random.rand(8784) * 100,
            'temp_air': np.random.rand(8784) * 20,
            'wind_speed': np.random.rand(8784) * 10,
        }, index=idx)

        cleaned = dm._remove_leap_day(df)
        assert len(cleaned) == 8760
        assert not ((cleaned.index.month == 2) & (cleaned.index.day == 29)).any()

    def test_invalid_year_raises(self, tmp_path):
        dm = DataManager(cache_dir=str(tmp_path / 'cache'), raw_dir=str(tmp_path / 'raw'))
        with pytest.raises(ValueError):
            dm.get_weather_data(39.0, 102.0, 1800)

    def test_download_pvgis_success(self, tmp_path, sample_weather_df):
        dm = DataManager(cache_dir=str(tmp_path / 'cache'), raw_dir=str(tmp_path / 'raw'))

        mock_data = sample_weather_df.rename(columns={
            'ghi': 'G(h)', 'dni': 'Gb(n)', 'dhi': 'Gd(h)',
            'temp_air': 'T2m', 'wind_speed': 'WS10m'
        })

        with patch('src.data_manager.pvlib.iotools.get_pvgis_hourly',
                   return_value=(mock_data, {})) as mock_get:
            df = dm.get_weather_data(39.0, 102.0, 2020, use_cache=False)
            assert all(col in df.columns for col in REQUIRED_COLUMNS)
            mock_get.assert_called_once()

    def test_corrupt_cache_deleted_and_redownloaded(self, tmp_path, sample_weather_df):
        cache_dir = tmp_path / 'cache'
        raw_dir = tmp_path / 'raw'
        dm = DataManager(cache_dir=str(cache_dir), raw_dir=str(raw_dir))

        # 写入缺少列的坏缓存
        bad_df = sample_weather_df.drop(columns=['dni'])
        dm.save_raw_data(bad_df, 39.0, 102.0, 2020)

        mock_data = sample_weather_df.rename(columns={
            'ghi': 'G(h)', 'dni': 'Gb(n)', 'dhi': 'Gd(h)',
            'temp_air': 'T2m', 'wind_speed': 'WS10m'
        })

        with patch('src.data_manager.pvlib.iotools.get_pvgis_hourly',
                   return_value=(mock_data, {})):
            df = dm.get_weather_data(39.0, 102.0, 2020, use_cache=True)
            assert 'dni' in df.columns
