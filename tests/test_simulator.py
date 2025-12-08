import pytest
import pandas as pd
import numpy as np
from src.simulator import PVSimulator

@pytest.fixture
def mock_weather_data():
    """生成模拟的晴天数据"""
    times = pd.date_range(start='2021-01-01', periods=24, freq='H', tz='UTC')
    df = pd.DataFrame(index=times)
    
    # 简单的钟形曲线模拟 GHI
    hour = times.hour
    ghi = np.maximum(0, 1000 * np.sin(np.pi * (hour - 6) / 12))
    ghi[(hour < 6) | (hour > 18)] = 0
    
    df['ghi'] = ghi
    df['dni'] = ghi * 0.8 # 假设
    df['dhi'] = ghi * 0.2
    df['temp_air'] = 25
    df['wind_speed'] = 2
    return df

def test_simulator_run(mock_weather_data):
    # 1MW 系统
    sim = PVSimulator(lat=30, lon=120, capacity_mw=1.0)
    results = sim.run(mock_weather_data)
    
    # 检查是否有结果
    assert results.ac is not None
    assert len(results.ac) == 24
    
    # 检查白天是否有出力
    noon_power = results.ac.iloc[12] # 中午12点
    assert noon_power > 0
    
    # 检查夜间是否无出力
    night_power = results.ac.iloc[0] # 午夜0点
    assert night_power <= 0.1 # 浮点误差
