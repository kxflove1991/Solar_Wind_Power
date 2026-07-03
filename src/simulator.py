import pvlib
from pvlib.pvsystem import PVSystem, Array, FixedMount
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class PVSimulator:
    def __init__(self, lat, lon, capacity_mw=1.0, surface_tilt=None, surface_azimuth=180, module_name=None, inverter_name=None):
        """
        初始化光伏模拟器
        :param lat: 纬度
        :param lon: 经度
        :param capacity_mw: 系统容量 (MW)
        :param surface_tilt: 倾角 (默认为纬度)
        :param surface_azimuth: 方位角 (180=南)
        :param module_name: 组件型号 (默认 None, 使用简化模型)
        :param inverter_name: 逆变器型号 (默认 None, 使用简化模型)
        """
        self.location = Location(lat, lon, tz='UTC') # PVGIS数据通常为UTC
        self.capacity_w = capacity_mw * 1e6 # MW -> W
        
        # 默认最佳倾角约为纬度
        if surface_tilt is None:
            surface_tilt = abs(lat)
            
        self.surface_tilt = surface_tilt
        self.surface_azimuth = surface_azimuth
        
        self.module_name = module_name
        self.inverter_name = inverter_name
        
        self.mc = None
        self._setup_model()

    def _setup_model(self):
        # 1. 定义支架
        mount = FixedMount(surface_tilt=self.surface_tilt, surface_azimuth=self.surface_azimuth)
        
        # 2. 组件参数
        if self.module_name:
            # 尝试从 Sandia 数据库获取组件参数
            sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
            module_params = sandia_modules[self.module_name]
            temperature_model_parameters = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
        else:
            # 默认 PVWatts 简化模型
            # gamma_pdc: 温度系数 (-0.4%/C)
            module_params = {
                'pdc0': self.capacity_w, 
                'gamma_pdc': -0.004
            }
            temperature_model_parameters = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
        
        # 3. 逆变器参数
        if self.inverter_name:
            # 尝试从 CEC 数据库获取逆变器参数
            cec_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
            inverter_params = cec_inverters[self.inverter_name]
        else:
            # 默认简化模型
            # 假设容配比 1.2, 逆变器效率 96%
            inverter_params = {
                'pdc0': self.capacity_w / 1.2, 
                'eta_inv_nom': 0.96
            }
        
        # 4. 构建系统
        array = Array(
            mount=mount, 
            module_parameters=module_params, 
            temperature_model_parameters=temperature_model_parameters
        )
        system = PVSystem(arrays=[array], inverter_parameters=inverter_params)
        
        # 5. 构建模型链
        self.mc = ModelChain(
            system, 
            self.location, 
            aoi_model='physical', 
            spectral_model='no_loss', 
            losses_model='pvwatts'
        )
        logger.info("PVSystem 初始化完成")

    def run(self, weather_data):
        """
        执行模拟
        :param weather_data: 包含 ghi, dni, dhi, temp_air, wind_speed 的 DataFrame
        """
        logger.info("开始光伏出力模拟...")
        try:
            self.mc.run_model(weather_data)
            return self.mc.results
        except Exception as e:
            logger.error(f"模拟失败: {e}")
            raise
