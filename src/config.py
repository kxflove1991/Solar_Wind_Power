import argparse
import logging
import sys
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SimulationConfig:
    """模拟配置数据类"""
    lat: float = 39.0
    lon: float = 102.0
    year: int = 2020
    # 容量：推荐分别指定光伏和风电容量
    pv_capacity_mw: float = 100.0
    wind_capacity_mw: float = 100.0
    # 兼容旧版：capacity_mw 保留，作为 pv/wind 的默认值
    capacity_mw: Optional[float] = None

    hub_height: int = 120
    turbine_type: str = 'custom_10mw'
    surface_tilt: Optional[float] = None
    surface_azimuth: float = 180
    module_name: Optional[str] = None
    inverter_name: Optional[str] = None
    database: str = 'PVGIS-ERA5'
    output_dir: str = 'data/processed'
    figures_dir: str = 'figures'
    reports_dir: str = 'reports'
    log_level: str = 'INFO'

    def __post_init__(self):
        """校验配置并兼容旧参数"""
        if not (-90 <= self.lat <= 90):
            raise ValueError(f"lat 必须在 [-90, 90] 之间，得到 {self.lat}")
        if not (-180 <= self.lon <= 180):
            raise ValueError(f"lon 必须在 [-180, 180] 之间，得到 {self.lon}")
        if self.year < 2000 or self.year > 2030:
            raise ValueError(f"year 超出合理范围 [2000, 2030]，得到 {self.year}")
        if self.hub_height <= 0:
            raise ValueError(f"hub_height 必须为正，得到 {self.hub_height}")
        if self.surface_azimuth < 0 or self.surface_azimuth > 360:
            raise ValueError(f"surface_azimuth 必须在 [0, 360] 之间，得到 {self.surface_azimuth}")
        if self.capacity_mw is not None:
            if self.capacity_mw <= 0:
                raise ValueError(f"capacity 必须为正，得到 {self.capacity_mw}")
            self.pv_capacity_mw = self.capacity_mw
            self.wind_capacity_mw = self.capacity_mw
        if self.pv_capacity_mw <= 0:
            raise ValueError(f"pv_capacity_mw 必须为正，得到 {self.pv_capacity_mw}")
        if self.wind_capacity_mw <= 0:
            raise ValueError(f"wind_capacity_mw 必须为正，得到 {self.wind_capacity_mw}")
        if self.turbine_type not in SUPPORTED_TURBINES:
            raise ValueError(
                f"turbine_type 不支持: {self.turbine_type}. 支持的类型: {SUPPORTED_TURBINES}"
            )
        if self.database not in SUPPORTED_DATABASES:
            raise ValueError(
                f"database 不支持: {self.database}. 支持的数据库: {SUPPORTED_DATABASES}"
            )


SUPPORTED_TURBINES = ('custom_10mw', 'generic_2mw')
SUPPORTED_DATABASES = ('PVGIS-ERA5',)


def parse_args(argv: Optional[list] = None) -> SimulationConfig:
    """解析命令行参数并返回配置对象"""
    parser = argparse.ArgumentParser(
        description='光伏与风电系统全流程分析程序',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main_analysis.py
  python main_analysis.py --lat 40.0 --lon 116.0 --year 2019 --pv-capacity 50 --wind-capacity 100
  python main_analysis.py --lat 39.0 --lon 102.0 --year 2020 --capacity 100 --hub-height 140
        """
    )

    parser.add_argument('--lat', type=float, default=39.0, help='站点纬度 (默认: 39.0)')
    parser.add_argument('--lon', type=float, default=102.0, help='站点经度 (默认: 102.0)')
    parser.add_argument('--year', type=int, default=2020, help='分析年份 (默认: 2020)')

    # 容量：新版推荐分别指定；旧版 --capacity 仍保留
    parser.add_argument('--capacity', type=float, default=None,
                        help='兼容旧版：统一装机容量 (MW)，当指定时 pv/wind 均使用此值')
    parser.add_argument('--pv-capacity', type=float, default=100.0, help='光伏装机容量 (MW) (默认: 100.0)')
    parser.add_argument('--wind-capacity', type=float, default=100.0, help='风电装机容量 (MW) (默认: 100.0)')

    parser.add_argument('--surface-tilt', type=float, default=None,
                        help='光伏组件倾角 (默认: 纬度绝对值)')
    parser.add_argument('--surface-azimuth', type=float, default=180,
                        help='光伏组件方位角 (默认: 180, 正南)')
    parser.add_argument('--module-name', type=str, default=None,
                        help='光伏组件型号 (默认: 使用 PVWatts 简化模型)')
    parser.add_argument('--inverter-name', type=str, default=None,
                        help='逆变器型号 (默认: 使用简化模型)')

    parser.add_argument('--hub-height', type=int, default=120, help='风机轮毂高度 (m) (默认: 120)')
    parser.add_argument('--turbine-type', type=str, default='custom_10mw', help='风机型号')

    parser.add_argument('--database', type=str, default='PVGIS-ERA5', help='气象数据库')

    parser.add_argument('--output-dir', type=str, default='data/processed', help='处理数据输出目录')
    parser.add_argument('--figures-dir', type=str, default='figures', help='图表输出目录')
    parser.add_argument('--reports-dir', type=str, default='reports', help='报告输出目录')

    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='日志级别 (默认: INFO)')

    args = parser.parse_args(argv)

    try:
        return SimulationConfig(
            lat=args.lat,
            lon=args.lon,
            year=args.year,
            capacity_mw=args.capacity,
            pv_capacity_mw=args.pv_capacity,
            wind_capacity_mw=args.wind_capacity,
            hub_height=args.hub_height,
            turbine_type=args.turbine_type,
            surface_tilt=args.surface_tilt,
            surface_azimuth=args.surface_azimuth,
            module_name=args.module_name,
            inverter_name=args.inverter_name,
            database=args.database,
            output_dir=args.output_dir,
            figures_dir=args.figures_dir,
            reports_dir=args.reports_dir,
            log_level=args.log_level
        )
    except ValueError as e:
        parser.error(str(e))


def setup_logging(log_level: str = 'INFO'):
    """配置全局日志"""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
