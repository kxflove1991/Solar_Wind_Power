import pandas as pd
import numpy as np
import logging
from pathlib import Path
from windpowerlib import ModelChain, WindTurbine

logger = logging.getLogger(__name__)

DEFAULT_ROUGHNESS_LENGTH = 0.05
DEFAULT_PRESSURE = 101325.0
TURBINE_DATA_DIR = Path(__file__).parent.parent / 'data' / 'turbines'


def _load_power_curve(curve_name: str) -> pd.DataFrame:
    """从 CSV 加载功率曲线"""
    path = TURBINE_DATA_DIR / f"{curve_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"找不到功率曲线文件: {path}")
    df = pd.read_csv(path, comment='#')
    required = {'wind_speed', 'power'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"功率曲线文件缺少列: {missing}")
    return df.rename(columns={'power': 'value'})


class WindSimulator:
    """风电出力模拟器"""

    def __init__(
        self,
        lat: float,
        lon: float,
        capacity_mw: float = 1.0,
        hub_height: int = 120,
        turbine_type: str = 'custom_10mw',
        use_integer_turbines: bool = False,
    ):
        self.lat = lat
        self.lon = lon
        self.capacity_mw = capacity_mw
        self.hub_height = hub_height
        self.turbine_type = turbine_type
        self.use_integer_turbines = use_integer_turbines

        logger.info(
            f"初始化风电模拟器: 机型={turbine_type}, 轮毂高度={hub_height}m, "
            f"目标容量={capacity_mw}MW"
        )

        self.turbine = self._build_turbine(turbine_type, hub_height)
        self.mc = ModelChain(
            self.turbine,
            wind_speed_model='logarithmic',
            temperature_model='linear_gradient',
            density_model='ideal_gas',
            power_output_model='power_curve'
        )

    def _build_turbine(self, turbine_type: str, hub_height: int) -> WindTurbine:
        if turbine_type == 'custom_10mw':
            logger.info("使用自定义 10MW 风机模型 (适用于中国西北地区大型风机)")
            curve = _load_power_curve('custom_10mw')
            return WindTurbine(
                hub_height=hub_height,
                nominal_power=10_000_000,
                power_curve=curve,
                power_coefficient_curve=None
            )
        elif turbine_type == 'generic_2mw':
            logger.info("使用通用 2MW 风机模型")
            curve = _load_power_curve('generic_2mw')
            return WindTurbine(
                hub_height=hub_height,
                nominal_power=2_000_000,
                power_curve=curve,
                power_coefficient_curve=None
            )
        else:
            raise ValueError(f"不支持的风机类型: {turbine_type}")

    def _scale_power(self, single_turbine_power: pd.Series) -> pd.Series:
        """将单机功率缩放为目标容量"""
        nominal = self.turbine.nominal_power
        if not nominal or nominal <= 0:
            return single_turbine_power

        scale_factor = (self.capacity_mw * 1e6) / nominal

        if self.use_integer_turbines:
            n_turbines = max(1, round(scale_factor))
            final_power = single_turbine_power * n_turbines
            logger.info(f"使用整数风机数量: {n_turbines} 台")
        else:
            final_power = single_turbine_power * scale_factor

        return final_power.clip(lower=0)

    def run(self, weather_df: pd.DataFrame) -> pd.Series:
        """运行风电模拟"""
        logger.info("开始风电出力模拟...")

        required = {'wind_speed', 'temp_air'}
        missing = required - set(weather_df.columns)
        if missing:
            raise ValueError(f"气象数据缺少必要列: {missing}")

        df = pd.DataFrame(index=weather_df.index)
        df[('wind_speed', 10)] = weather_df['wind_speed']
        df[('temperature', 2)] = weather_df['temp_air']
        df[('roughness_length', 0)] = weather_df.get('roughness_length', DEFAULT_ROUGHNESS_LENGTH)
        df[('pressure', 0)] = weather_df.get('pressure', DEFAULT_PRESSURE)
        df.columns = pd.MultiIndex.from_tuples(df.columns, names=['variable_name', 'height'])

        self.mc.run_model(df)
        power_w = self.mc.power_output

        if power_w is None:
            raise RuntimeError("风电模拟失败: power_output 为 None")
        if len(power_w) != len(weather_df):
            raise RuntimeError(f"风电模拟输出长度不匹配: 期望 {len(weather_df)}, 实际 {len(power_w)}")
        if power_w.isna().all():
            raise RuntimeError("风电模拟结果全部为 NaN")

        final_power = self._scale_power(power_w)
        logger.info(f"风电模拟完成。总发电量: {final_power.sum() / 1e6:.2f} MWh")
        return final_power
