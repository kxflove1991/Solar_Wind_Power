import unittest
import pandas as pd
import numpy as np
import shutil
from pathlib import Path
from src.data_manager import DataManager

class TestDataManager(unittest.TestCase):
    def setUp(self):
        # 使用临时缓存目录
        self.test_cache_dir = Path("tests/temp_cache")
        self.dm = DataManager(cache_dir=self.test_cache_dir)
        self.lat = 30.0
        self.lon = 100.0
        
    def tearDown(self):
        # 清理临时目录
        if self.test_cache_dir.exists():
            shutil.rmtree(self.test_cache_dir)

    def test_leap_year_handling(self):
        """测试闰年数据处理：应自动移除 2月29日"""
        year = 2020
        # 1. 创建模拟的闰年数据 (8784 小时)
        dates = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31 23:00', freq='h')
        self.assertEqual(len(dates), 8784)
        
        df = pd.DataFrame(index=dates, data={
            'ghi': np.random.rand(8784) * 1000,
            'dni': np.random.rand(8784) * 1000, 
            'dhi': np.random.rand(8784) * 1000,
            'temp_air': np.random.rand(8784) * 30
        })
        
        # 写入缓存 (模拟已下载状态)
        cache_file = self.test_cache_dir / f"weather_{self.lat:.4f}_{self.lon:.4f}_{year}_PVGIS-ERA5.csv"
        df.to_csv(cache_file)
        
        # 2. 调用 DataManager 读取
        data = self.dm.get_weather_data(self.lat, self.lon, year)
        
        # 3. 验证
        self.assertEqual(len(data), 8760, "闰年数据应被裁剪至 8760 小时")
        # 验证 2月29日 是否存在
        leap_day_data = data[(data.index.month == 2) & (data.index.day == 29)]
        self.assertTrue(leap_day_data.empty, "2月29日的数据应该被移除")

    def test_normal_year_handling(self):
        """测试平年数据处理：应保持 8760 小时"""
        year = 2019
        dates = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31 23:00', freq='h')
        self.assertEqual(len(dates), 8760)
        
        df = pd.DataFrame(index=dates, data={
            'ghi': np.random.rand(8760),
            'dni': np.random.rand(8760),
            'dhi': np.random.rand(8760),
            'temp_air': np.random.rand(8760)
        })
        
        cache_file = self.test_cache_dir / f"weather_{self.lat:.4f}_{self.lon:.4f}_{year}_PVGIS-ERA5.csv"
        df.to_csv(cache_file)
        
        data = self.dm.get_weather_data(self.lat, self.lon, year)
        
        self.assertEqual(len(data), 8760)

if __name__ == '__main__':
    unittest.main()
