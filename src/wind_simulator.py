import pandas as pd
import numpy as np
import logging
from windpowerlib import ModelChain, WindTurbine

logger = logging.getLogger(__name__)

class WindSimulator:
    def __init__(self, lat, lon, capacity_mw=1.0, hub_height=120, turbine_type='custom_10mw'):
        """
        初始化风电模拟器
        :param lat: 纬度
        :param lon: 经度
        :param capacity_mw: 目标装机容量 (MW)
        :param hub_height: 轮毂高度 (m), 默认 120m
        :param turbine_type: 风机型号 (默认 'custom_10mw')
        """
        self.lat = lat
        self.lon = lon
        self.capacity_mw = capacity_mw
        self.hub_height = hub_height
        self.turbine_type = turbine_type
        
        logger.info(f"初始化风电模拟器: 机型={turbine_type}, 轮毂高度={hub_height}m, 目标容量={capacity_mw}MW")
        
        if turbine_type == 'custom_10mw':
            # 自定义 10MW 风机功率曲线 (参考通用 10MW 级风机特性)
            # Cut-in: ~3m/s, Rated: ~11m/s, Cut-out: 25m/s
            logger.info("使用自定义 10MW 风机模型 (适用于中国西北地区大型风机)")
            
            # 构建完整的风速-功率点 (0-25 m/s)
            wind_speeds = [0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
            power_values = [0.0, 0.0, 150_000.0, 650_000.0, 1_200_000.0, 2_100_000.0, 
                           3_300_000.0, 5_000_000.0, 7_000_000.0, 8_800_000.0, 10_000_000.0, 10_000_000.0]
            
            # 补充 13-24 m/s 的额定功率点
            for ws in range(13, 25):
                wind_speeds.append(float(ws))
                power_values.append(10_000_000.0)
            
            # 添加切出风速点
            wind_speeds.extend([25.0, 25.1])
            power_values.extend([10_000_000.0, 0.0])
            
            self.turbine = WindTurbine(
                hub_height=hub_height,
                nominal_power=10_000_000, # 10MW
                power_curve=pd.DataFrame(
                    data={
                        'value': power_values,
                        'wind_speed': wind_speeds
                    }
                ),
                power_coefficient_curve=None
            )
        else:
            try:
                # 尝试从 windpowerlib 内置数据库加载风机数据
                self.turbine = WindTurbine(
                    hub_height=hub_height,
                    turbine_type=turbine_type
                )
            except Exception as e:
                logger.warning(f"无法加载指定风机数据 ({e})，将使用通用 2MW 风机参数")
                # 备用：定义一个通用的 2MW 风机
                self.turbine = WindTurbine(
                    hub_height=hub_height,
                    nominal_power=2000000, # 2MW
                    power_curve=pd.DataFrame(
                        data={'value': [0.0, 0.0, 50000.0, 200000.0, 500000.0, 1000000.0, 1500000.0, 1900000.0, 2000000.0, 2000000.0],
                          'wind_speed': [0.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 25.0]}
                ),
                power_coefficient_curve=None
                )

        # 初始化模型链
        # 改用 logarithmic (对数律) 模型，更适合配合粗糙度长度使用
        self.mc = ModelChain(
            self.turbine, 
            wind_speed_model='logarithmic',
            temperature_model='linear_gradient',
            density_model='ideal_gas',
            power_output_model='power_curve'
        )

    def run(self, weather_df):
        """
        运行风电模拟
        :param weather_df: 必须包含 'wind_speed' (m/s), 'temp_air' (C)
        :return: Series (功率, 单位 W)
        """
        logger.info("开始风电出力模拟...")
        
        # 准备 windpowerlib 需要的数据格式 (MultiIndex)
        df = pd.DataFrame(index=weather_df.index)
        
        # 1. 风速 (假设原始数据在 10m 高度)
        df[('wind_speed', 10)] = weather_df['wind_speed']
        
        # 2. 温度 (假设原始数据在 2m 高度)
        df[('temperature', 2)] = weather_df['temp_air']
        
        # 3. 粗糙度 (如果数据中没有，设为 0.05 - 适用于开阔平坦地形/荒漠草原)
        if 'roughness_length' in weather_df.columns:
            df[('roughness_length', 0)] = weather_df['roughness_length']
        else:
            df[('roughness_length', 0)] = 0.05
            
        # 4. 气压 (如果数据中没有，设为标准大气压 101325 Pa)
        # windpowerlib 默认可能需要该列来计算密度
        if 'pressure' in weather_df.columns:
            df[('pressure', 0)] = weather_df['pressure']
        else:
            df[('pressure', 0)] = 101325.0

        # 转换为 MultiIndex 列
        df.columns = pd.MultiIndex.from_tuples(df.columns, names=['variable_name', 'height'])
        
        # 运行模拟
        self.mc.run_model(df)
        
        # 获取结果 (单机功率 W)
        power_w = self.mc.power_output
        
        # 验证结果
        if power_w is None:
            raise RuntimeError("风电模拟失败: power_output 为 None")
        if len(power_w) != len(weather_df):
            raise RuntimeError(f"风电模拟输出长度不匹配: 期望 {len(weather_df)}, 实际 {len(power_w)}")
        if power_w.isna().all():
            raise RuntimeError("风电模拟结果全部为 NaN")
        
        # 缩放到目标容量
        if self.turbine.nominal_power:
            scale_factor = (self.capacity_mw * 1e6) / self.turbine.nominal_power
            final_power = power_w * scale_factor
        else:
            # 如果没有 nominal_power (不太可能)，直接返回
            final_power = power_w
            
        # 确保没有负值
        final_power = final_power.clip(lower=0)
        
        logger.info(f"风电模拟完成。总发电量: {final_power.sum()/1e6:.2f} MWh")
        
        return final_power
