import os
import pvlib
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataManager:
    def __init__(self, cache_dir='cache', raw_dir='data/raw/weather'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def save_raw_data(self, df, lat, lon):
        """保存原始气象数据 (符合用户要求的格式)"""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"weather_{timestamp}.csv"
        filepath = self.raw_dir / filename
        
        # 添加元数据头
        metadata = f"# Location: Lat={lat}, Lon={lon}\n# Created: {timestamp}\n# Source: PVGIS-ERA5\n"
        
        try:
            with open(filepath, 'w') as f:
                f.write(metadata)
                df.to_csv(f) # index is timestamp
            logging.info(f"原始气象数据已保存: {filepath}")
            return filepath
        except Exception as e:
            logging.error(f"保存原始数据失败: {e}")
            return None

    def get_weather_data(self, lat, lon, year, database='PVGIS-ERA5'):
        """
        获取气象数据 (优先读取缓存)
        :param lat: 纬度
        :param lon: 经度
        :param year: 年份
        :param database: 数据库 (默认 PVGIS-ERA5)
        :return: DataFrame (标准化列名)
        """
        # 1. 检查缓存
        # 文件名包含关键参数
        cache_file = self.cache_dir / f"weather_{lat:.4f}_{lon:.4f}_{year}_{database}.csv"
        
        if cache_file.exists():
            logging.info(f"命中缓存: {cache_file}")
            # 读取缓存时需解析时间索引
            try:
                data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                # 确保时区信息 (CSV读取后通常丢失时区，设为UTC)
                if data.index.tz is None:
                    data.index = data.index.tz_localize('UTC')
                return data
            except Exception as e:
                logging.warning(f"缓存读取失败，将重新下载: {e}")

        # 2. 下载数据
        logging.info(f"开始下载: Lat={lat}, Lon={lon}, Year={year}, DB={database}")
        try:
            # get_pvgis_hourly 返回值可能随版本不同
            # 通常为 (data, inputs, metadata) 或 (data, metadata)
            res = pvlib.iotools.get_pvgis_hourly(
                latitude=lat,
                longitude=lon,
                start=year,
                end=year,
                raddatabase=database,
                components=True,
                url='https://re.jrc.ec.europa.eu/api/v5_2/'
            )
            
            if len(res) == 3:
                data, inputs, metadata = res
            elif len(res) == 2:
                data, metadata = res
                inputs = {} # 默认为空
            else:
                data = res[0]
            
            # Debug: 打印列名
            logging.info(f"PVGIS 返回列名: {data.columns.tolist()}")

            # 3. 标准化处理
            # PVGIS 返回列名可能为: 'poa_global', 'bi', 'di', 'ri' 等
            # 或者 'G(h)', 'Gb(n)', 'Gd(h)'
            rename_map = {
                'T2m': 'temp_air', 
                'WS10m': 'wind_speed',
                'poa_global': 'ghi', # 假设水平面
                'Gb(n)': 'dni',
                'Gd(h)': 'dhi',
                'G(h)': 'ghi',
            }
            data = data.rename(columns={k: v for k, v in rename_map.items() if k in data.columns})

            # 4. 如果缺少 GHI/DNI/DHI，尝试从 POA 重构 (假设 horizontal)
            if 'ghi' not in data.columns and 'poa_direct' in data.columns:
                logging.info("通过 POA 分量重构 GHI/DNI/DHI...")
                data['ghi'] = data['poa_direct'] + data['poa_sky_diffuse'] + data['poa_ground_diffuse']
                data['dhi'] = data['poa_sky_diffuse']
                
                # 计算 DNI = Beam_Horiz / cos(Zenith)
                loc = pvlib.location.Location(lat, lon)
                solpos = loc.get_solarposition(data.index)
                zenith = solpos['apparent_zenith']
                
                # 避免除以零 (cos(85) ~ 0.087)
                cos_z = np.cos(np.radians(zenith))
                data['dni'] = data['poa_direct'] / np.maximum(cos_z, 0.01) # 限制最小值
                
                # 夜间修正
                data.loc[zenith > 85, 'dni'] = 0
                data.loc[zenith > 85, 'ghi'] = 0
                data.loc[zenith > 85, 'dhi'] = 0

            # 简单校验
            required = ['ghi', 'dni', 'dhi', 'temp_air']
            
            # 简单校验
            required = ['ghi', 'dni', 'dhi', 'temp_air']
            missing = [col for col in required if col not in data.columns]
            if missing:
                logging.warning(f"警告: 数据缺少必要列: {missing}，可能影响后续计算")

            # 4. 写入缓存
            data.to_csv(cache_file)
            logging.info(f"数据已缓存至: {cache_file}")
            
            return data
            
        except Exception as e:
            logging.error(f"数据获取失败: {e}")
            raise
