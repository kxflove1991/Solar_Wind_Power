import pandas as pd
import numpy as np

SUNSHINE_THRESHOLD = 120  # W/m²，WMO 定义的有效日照阈值
WH_TO_KWH = 1000
REQUIRED_COLUMNS = ['ghi', 'dni', 'dhi', 'temp_air']


class ResourceAnalyzer:
    """气象与辐射资源分析器"""

    def __init__(self, weather_df: pd.DataFrame):
        if not isinstance(weather_df.index, pd.DatetimeIndex):
            raise TypeError("weather_df 必须具有 DatetimeIndex")
        missing = [c for c in REQUIRED_COLUMNS if c not in weather_df.columns]
        if missing:
            raise ValueError(f"weather_df 缺少必要列: {missing}")
        self.df = weather_df

    def analyze_radiation(self) -> dict:
        """基础辐射统计"""
        stats = self.df[['ghi', 'dni', 'dhi']].describe()
        sunshine_hours = int((self.df['ghi'] > SUNSHINE_THRESHOLD).sum())
        yearly_irradiation = self.df[['ghi', 'dni', 'dhi']].sum() / WH_TO_KWH
        avg_temp = float(self.df['temp_air'].mean())

        return {
            'stats': stats,
            'sunshine_hours': sunshine_hours,
            'yearly_irradiation': yearly_irradiation,
            'avg_temp': avg_temp
        }

    def analyze_seasonality(self) -> pd.DataFrame:
        """季节性分析 (按月聚合)"""
        df_copy = self.df.copy()
        df_copy['month'] = df_copy.index.month

        monthly_stats = df_copy.groupby('month').agg({
            'ghi': 'sum',
            'dni': 'sum',
            'temp_air': 'mean'
        })

        monthly_stats['ghi'] = monthly_stats['ghi'] / WH_TO_KWH
        monthly_stats['dni'] = monthly_stats['dni'] / WH_TO_KWH
        return monthly_stats
