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
    capacity_mw: float = 100.0
    hub_height: int = 120
    turbine_type: str = 'custom_10mw'
    surface_tilt: Optional[float] = None
    surface_azimuth: float = 180
    module_name: Optional[str] = None
    inverter_name: Optional[str] = None
    database: str = 'PVGIS-ERA5'
    output_dir: str = 'data/processed/pv_normalized'
    figures_dir: str = 'figures'
    reports_dir: str = 'reports'
    log_level: str = 'INFO'


def parse_args() -> SimulationConfig:
    """解析命令行参数并返回配置对象"""
    parser = argparse.ArgumentParser(
        description='光伏与风电系统全流程分析程序',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main_analysis.py
  python main_analysis.py --lat 40.0 --lon 116.0 --year 2019 --capacity 50
  python main_analysis.py --lat 39.0 --lon 102.0 --year 2020 --capacity 100 --hub-height 140
        """
    )

    # 站点参数
    parser.add_argument('--lat', type=float, default=39.0,
                        help='站点纬度 (默认: 39.0)')
    parser.add_argument('--lon', type=float, default=102.0,
                        help='站点经度 (默认: 102.0)')
    parser.add_argument('--year', type=int, default=2020,
                        help='分析年份 (默认: 2020)')

    # 系统容量
    parser.add_argument('--capacity', type=float, default=100.0,
                        help='系统装机容量 (MW) (默认: 100.0)')

    # 光伏参数
    parser.add_argument('--surface-tilt', type=float, default=None,
                        help='光伏组件倾角 (默认: 纬度绝对值)')
    parser.add_argument('--surface-azimuth', type=float, default=180,
                        help='光伏组件方位角 (默认: 180, 正南)')
    parser.add_argument('--module-name', type=str, default=None,
                        help='光伏组件型号 (默认: 使用PVWatts简化模型)')
    parser.add_argument('--inverter-name', type=str, default=None,
                        help='逆变器型号 (默认: 使用简化模型)')

    # 风电参数
    parser.add_argument('--hub-height', type=int, default=120,
                        help='风机轮毂高度 (m) (默认: 120)')
    parser.add_argument('--turbine-type', type=str, default='custom_10mw',
                        help='风机型号 (默认: custom_10mw)')

    # 数据源
    parser.add_argument('--database', type=str, default='PVGIS-ERA5',
                        help='气象数据库 (默认: PVGIS-ERA5)')

    # 输出目录
    parser.add_argument('--output-dir', type=str, default='data/processed/pv_normalized',
                        help='处理数据输出目录')
    parser.add_argument('--figures-dir', type=str, default='figures',
                        help='图表输出目录')
    parser.add_argument('--reports-dir', type=str, default='reports',
                        help='报告输出目录')

    # 日志
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='日志级别 (默认: INFO)')

    args = parser.parse_args()

    return SimulationConfig(
        lat=args.lat,
        lon=args.lon,
        year=args.year,
        capacity_mw=args.capacity,
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


def setup_logging(log_level: str = 'INFO'):
    """配置全局日志"""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
