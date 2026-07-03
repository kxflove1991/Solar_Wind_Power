import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import pvlib
from typing import Optional

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ['ghi', 'dni', 'dhi', 'temp_air', 'wind_speed']
PVGIS_START_YEAR = 2005
PVGIS_END_YEAR = 2023


class DataManager:
    """
    负责从 PVGIS 获取气象数据、缓存、清洗和本地持久化。
    """
    def __init__(
        self,
        cache_dir: str = 'cache',
        raw_dir: str = 'data/raw/weather',
    ):
        self.cache_dir = Path(cache_dir)
        self.raw_dir = Path(raw_dir)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def _cache_file(self, lat: float, lon: float, year: int) -> Path:
        return self.cache_dir / f"weather_{lat:.4f}_{lon:.4f}_{year}_PVGIS-ERA5.csv"

    def _raw_file(self, lat: float, lon: float, year: int) -> Path:
        return self.raw_dir / f"weather_{lat:.4f}_{lon:.4f}_{year}.csv"

    def get_weather_data(
        self,
        lat: float,
        lon: float,
        year: int,
        database: str = 'PVGIS-ERA5',
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        获取指定经纬度和年份的逐时气象数据。

        返回的 DataFrame 索引为 UTC 时间，包含列：
        ['ghi', 'dni', 'dhi', 'temp_air', 'wind_speed']
        """
        self._validate_year(year)
        cache_file = self._cache_file(lat, lon, year)

        if use_cache and cache_file.exists():
            logger.info(f"从缓存加载气象数据: {cache_file}")
            df = self._load_from_csv(cache_file)
            if df is not None and self._has_required_columns(df):
                return df
            logger.warning("缓存文件损坏或列缺失，重新下载")
            try:
                cache_file.unlink()
            except OSError:
                pass

        logger.info(f"从 {database} 下载气象数据: lat={lat}, lon={lon}, year={year}")
        df = self._download_pvgis(lat, lon, year, database=database)
        self._validate_columns(df)

        # 保存缓存和原始文件
        self.save_raw_data(df, lat, lon, year)

        return df

    def _download_pvgis(self, lat: float, lon: float, year: int, database: str = 'PVGIS-ERA5') -> pd.DataFrame:
        if database != 'PVGIS-ERA5':
            raise ValueError(f"当前仅支持 PVGIS-ERA5 数据库，得到 {database}")

        try:
            result = pvlib.iotools.get_pvgis_hourly(
                lat,
                lon,
                start=year,
                end=year,
                rtd=True,
                map_variables=True,
                url='https://re.jrc.ec.europa.eu/api/v5_2/',
                timeout=30,
            )
            # pvlib >= 0.13.0 返回 (data, meta)；旧版返回 (data, inputs, metadata)
            data = result[0] if isinstance(result, tuple) else result
        except Exception as e:
            raise RuntimeError(f"PVGIS 数据下载失败: {e}") from e

        # PVGIS 返回 UTC 时间；保持 UTC，由上层负责时区转换
        if data.index.tz is None:
            data.index = data.index.tz_localize('UTC')

        # 处理闰年的 2 月 29 日：删除该日数据，保证 8760 小时
        data = self._remove_leap_day(data)

        data = self._rename_columns(data)
        return data

    def _remove_leap_day(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("输入数据必须是 DatetimeIndex")
        is_leap_feb29 = (df.index.month == 2) & (df.index.day == 29)
        if is_leap_feb29.any():
            logger.info(f"移除闰日数据，共 {is_leap_feb29.sum()} 条")
            df = df.loc[~is_leap_feb29]
        return df

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {
            'G(h)': 'ghi',
            'Gb(n)': 'dni',
            'Gd(h)': 'dhi',
            'T2m': 'temp_air',
            'WS10m': 'wind_speed',
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        return df

    def _has_required_columns(self, df: pd.DataFrame) -> bool:
        return all(col in df.columns for col in REQUIRED_COLUMNS)

    def _validate_columns(self, df: pd.DataFrame):
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"气象数据缺少必要列: {missing}")

    def _validate_year(self, year: int):
        if not isinstance(year, int):
            raise TypeError(f"year 必须是整数，得到 {type(year)}")
        if year < PVGIS_START_YEAR or year > PVGIS_END_YEAR:
            raise ValueError(
                f"PVGIS 仅支持 {PVGIS_START_YEAR}-{PVGIS_END_YEAR} 年，输入 year={year}"
            )

    def _load_from_csv(self, path: Path) -> Optional[pd.DataFrame]:
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True, comment='#')
            return df
        except Exception as e:
            logger.warning(f"读取 {path} 失败: {e}")
            return None

    def save_raw_data(self, df: pd.DataFrame, lat: float, lon: float, year: int):
        """保存原始气象数据到 raw_dir，并写入缓存"""
        raw_path = self._raw_file(lat, lon, year)
        cache_path = self._cache_file(lat, lon, year)

        metadata = (
            f"# Location: Lat={lat}, Lon={lon}\n"
            f"# Year: {year}\n"
            f"# Database: PVGIS-ERA5\n"
            f"# Generated: {datetime.now(timezone.utc).isoformat()}\n"
        )
        for path in (cache_path, raw_path):
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(metadata)
                df.to_csv(f)
