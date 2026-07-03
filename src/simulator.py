import pvlib
from pvlib.pvsystem import PVSystem, Array, FixedMount
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class PVSimulator:
    """光伏出力模拟器"""

    def __init__(
        self,
        lat: float,
        lon: float,
        capacity_mw: float = 1.0,
        surface_tilt: float = None,
        surface_azimuth: float = 180,
        module_name: str = None,
        inverter_name: str = None,
    ):
        self.location = Location(lat, lon, tz='UTC')
        self.capacity_w = capacity_mw * 1e6

        if surface_tilt is None:
            surface_tilt = abs(lat)

        self.surface_tilt = surface_tilt
        self.surface_azimuth = surface_azimuth
        self.module_name = module_name
        self.inverter_name = inverter_name
        self.mc = None
        self._setup_model()

    def _resolve_module_params(self):
        if not self.module_name:
            return {'pdc0': self.capacity_w, 'gamma_pdc': -0.004}

        sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
        if self.module_name not in sandia_modules:
            available = list(sandia_modules.index[:5])
            raise ValueError(
                f"组件 '{self.module_name}' 不在 SandiaMod 数据库中。"
                f"示例型号: {available}"
            )
        return sandia_modules[self.module_name]

    def _resolve_inverter_params(self):
        if not self.inverter_name:
            return {'pdc0': self.capacity_w / 1.2, 'eta_inv_nom': 0.96}

        cec_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
        if self.inverter_name not in cec_inverters:
            available = list(cec_inverters.index[:5])
            raise ValueError(
                f"逆变器 '{self.inverter_name}' 不在 CEC 数据库中。"
                f"示例型号: {available}"
            )
        return cec_inverters[self.inverter_name]

    def _setup_model(self):
        mount = FixedMount(surface_tilt=self.surface_tilt, surface_azimuth=self.surface_azimuth)

        module_params = self._resolve_module_params()
        inverter_params = self._resolve_inverter_params()

        temp_params = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

        array = Array(
            mount=mount,
            module_parameters=module_params,
            temperature_model_parameters=temp_params
        )
        system = PVSystem(arrays=[array], inverter_parameters=inverter_params)

        self.mc = ModelChain(
            system,
            self.location,
            aoi_model='physical',
            spectral_model='no_loss',
            losses_model='pvwatts'
        )
        logger.info("PVSystem 初始化完成")

    def run(self, weather_data: pd.DataFrame):
        """执行模拟"""
        logger.info("开始光伏出力模拟...")
        self.mc.run_model(weather_data)
        return self.mc.results
