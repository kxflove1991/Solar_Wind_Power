import pandas as pd
import numpy as np

class ResourceAnalyzer:
    def __init__(self, weather_df):
        self.df = weather_df

    def analyze_radiation(self):
        """
        基础辐射统计
        """
        # 基本统计量 (Min, Max, Mean, Std)
        stats = self.df[['ghi', 'dni', 'dhi']].describe()
        
        # 计算日照时长 (GHI > 120 W/m2 的小时数，WMO定义)
        sunshine_hours = (self.df['ghi'] > 120).sum()
        
        # 年累计辐射量 (kWh/m2)
        # 假设数据是每小时的，sum() 即为 Wh/m2，除以1000得到 kWh/m2
        yearly_irradiation = self.df[['ghi', 'dni', 'dhi']].sum() / 1000
        
        # 平均气温
        avg_temp = self.df['temp_air'].mean()
        
        return {
            'stats': stats,
            'sunshine_hours': int(sunshine_hours),
            'yearly_irradiation': yearly_irradiation,
            'avg_temp': avg_temp
        }

    def analyze_seasonality(self):
        """
        季节性分析 (按月聚合)
        """
        df_copy = self.df.copy()
        df_copy['Month'] = df_copy.index.month
        
        monthly_stats = df_copy.groupby('Month').agg({
            'ghi': 'sum',      # 月总辐射 (Wh/m2)
            'dni': 'sum',
            'temp_air': 'mean' # 月平均气温
        })
        
        # 转换为 kWh/m2
        monthly_stats['ghi'] = monthly_stats['ghi'] / 1000
        monthly_stats['dni'] = monthly_stats['dni'] / 1000
        
        return monthly_stats
