import pandas as pd
from src.data_manager import DataManager
from src.analyzer import ResourceAnalyzer
from src.simulator import PVSimulator
from src.visualizer import Visualizer
from src.quality_control import QualityControl
from src.wind_simulator import WindSimulator
from src.config import parse_args, setup_logging
import os
import logging
import sys
from datetime import datetime
from pathlib import Path


def main():
    # --- 1. 解析配置参数 ---
    config = parse_args()
    setup_logging(config.log_level)
    logger = logging.getLogger(__name__)

    logger.info(f"{'='*50}")
    logger.info("光伏与风电系统全流程分析程序 v1.3")
    logger.info(f"站点: Lat {config.lat}, Lon {config.lon}")
    logger.info(f"年份: {config.year}")
    logger.info(f"光伏容量: {config.pv_capacity_mw} MW, 风电容量: {config.wind_capacity_mw} MW")
    logger.info(f"{'='*50}")

    # 初始化 QC
    qc = QualityControl(output_dir=config.reports_dir)

    # 设置绘图样式
    Visualizer.set_style()

    try:
        # --- 2. 数据获取 ---
        logger.info(">> 步骤 1/5: 获取气象数据...")
        dm = DataManager()
        weather_df = dm.get_weather_data(
            config.lat, config.lon, config.year, database=config.database
        )
        logger.info(f"   [成功] 获取 {len(weather_df)} 小时数据")

        # QC 检查
        weather_df.attrs['lat'] = config.lat
        weather_df.attrs['lon'] = config.lon
        qc.check_integrity(weather_df, "Weather Data")
        qc.detect_outliers(weather_df, ['ghi', 'dni', 'temp_air'])

        # --- 3. 光资源分析与绘图 ---
        logger.info(">> 步骤 2/5: 分析光资源与绘图...")

        weather_df_plot = weather_df.copy()
        if weather_df_plot.index.tz is None:
            weather_df_plot.index = weather_df_plot.index.tz_localize('UTC')
        weather_df_plot.index = weather_df_plot.index.tz_convert('Asia/Shanghai')

        analyzer = ResourceAnalyzer(weather_df)
        rad_stats = analyzer.analyze_radiation()

        logger.info("   正在生成气象分析图表...")
        weather_fig_dir = Path(config.figures_dir) / "weather_plots"
        Visualizer.plot_weather_trends(weather_df_plot, weather_fig_dir)
        Visualizer.plot_weather_histograms(weather_df_plot, weather_fig_dir)
        Visualizer.plot_radiation_heatmaps(weather_df_plot, weather_fig_dir)
        Visualizer.plot_wind_analysis(weather_df_plot, weather_fig_dir)

        logger.info(f"   - 年总辐射 (GHI): {rad_stats['yearly_irradiation']['ghi']:.2f} kWh/m²")
        logger.info(f"   - 有效日照时数:   {rad_stats['sunshine_hours']} h")
        logger.info(f"   - 年平均气温:     {rad_stats['avg_temp']:.2f} °C")

        # --- 4. 光伏模拟 ---
        logger.info(">> 步骤 3/5: 模拟光伏出力...")
        sim = PVSimulator(
            config.lat, config.lon,
            capacity_mw=config.pv_capacity_mw,
            surface_tilt=config.surface_tilt,
            surface_azimuth=config.surface_azimuth,
            module_name=config.module_name,
            inverter_name=config.inverter_name
        )
        pv_results = sim.run(weather_df)
        logger.info("   [成功] 光伏模拟完成")

        # --- 4.5 风电模拟 ---
        logger.info(">> 步骤 3.5/5: 模拟风电出力...")
        wind_sim = WindSimulator(
            config.lat, config.lon,
            capacity_mw=config.wind_capacity_mw,
            hub_height=config.hub_height,
            turbine_type=config.turbine_type
        )
        wind_results = wind_sim.run(weather_df)
        logger.info("   [成功] 风电模拟完成")

        # 整理结果
        output = pd.DataFrame({
            'GHI': weather_df['ghi'],
            'AC_Power_W': pv_results.ac,
            'Wind_Power_W': wind_results
        }, index=weather_df.index)

        if output.index.tz is None:
            output.index = output.index.tz_localize('UTC')
        output.index = output.index.tz_convert('Asia/Shanghai')

        # --- 5. 数据处理与归一化 ---
        logger.info(">> 步骤 4/5: 处理标幺值数据...")
        output['PU'] = output['AC_Power_W'] / (config.pv_capacity_mw * 1e6)
        output['Wind_PU'] = output['Wind_Power_W'] / (config.wind_capacity_mw * 1e6)

        pu_save_dir = Path(config.output_dir)
        pu_save_dir.mkdir(parents=True, exist_ok=True)

        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_filename = f"pv_wind_normalized_{date_str}.csv"
        combined_path = pu_save_dir / combined_filename
        output.to_csv(combined_path)
        logger.info(f"   [保存] 标幺值数据已保存至 {combined_path}")

        qc.check_integrity(output, "PV & Wind Output Data")
        qc.detect_outliers(
            output,
            ['AC_Power_W', 'Wind_Power_W'],
            physical_bounds={
                'AC_Power_W': (0, config.pv_capacity_mw * 1e6),
                'Wind_Power_W': (0, config.wind_capacity_mw * 1e6),
            }
        )

        # 统计指标
        pv_total_mwh = output['AC_Power_W'].sum() / 1e6
        pv_util_hours = pv_total_mwh / config.pv_capacity_mw

        wind_total_mwh = output['Wind_Power_W'].sum() / 1e6
        wind_util_hours = wind_total_mwh / config.wind_capacity_mw

        logger.info(f"   [光伏] 年总发电量: {pv_total_mwh:.2f} MWh")
        logger.info(f"   [光伏] 等效利用小时: {pv_util_hours:.2f} h")
        logger.info(f"   [光伏] 容量系数: {pv_util_hours/8760:.2%}")
        logger.info(f"   [风电] 年总发电量: {wind_total_mwh:.2f} MWh")
        logger.info(f"   [风电] 等效利用小时: {wind_util_hours:.2f} h")
        logger.info(f"   [风电] 容量系数: {wind_util_hours/8760:.2%}")

        # --- 6. 生成图表报告 ---
        logger.info(">> 步骤 5/5: 生成图表报告...")

        pv_fig_dir = Path(config.figures_dir) / "pv_plots"
        wind_fig_dir = Path(config.figures_dir) / "wind_plots"

        # 串行绘图，避免 Matplotlib 多线程问题
        Visualizer.plot_pv_heatmap(output, pv_fig_dir)
        logger.info("   光伏出力分析图表生成完成")
        Visualizer.plot_pv_duration_curve(output, pv_fig_dir)
        Visualizer.plot_wind_power_analysis(output, wind_fig_dir)
        logger.info("   风电出力分析图表生成完成")

        qc_report_filename = "qc_report.txt"
        qc.generate_report(qc_report_filename)
        logger.info(f"   报告已保存至 {config.reports_dir}/{qc_report_filename}")

        logger.info("\n[完成] 所有任务已完成！")

    except Exception as e:
        logger.critical("程序执行过程中发生严重错误", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
