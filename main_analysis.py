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
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


def _plot_weather_charts(args):
    """绘制气象图表（用于线程池并发）"""
    df, save_dir = args
    Visualizer.plot_weather_trends(df, save_dir)
    Visualizer.plot_weather_duration_curves(df, save_dir)
    Visualizer.plot_radiation_heatmaps(df, save_dir)
    Visualizer.plot_wind_analysis(df, save_dir)


def _plot_pv_charts(args):
    """绘制光伏图表（用于线程池并发）"""
    df, save_dir = args
    Visualizer.plot_pv_heatmap(df, save_dir)
    Visualizer.plot_pv_duration_curve(df, save_dir)


def _plot_wind_charts(args):
    """绘制风电图表（用于线程池并发）"""
    df, save_dir = args
    Visualizer.plot_wind_power_analysis(df, save_dir)


def main():
    # --- 1. 解析配置参数 ---
    config = parse_args()
    setup_logging(config.log_level)
    logger = logging.getLogger(__name__)
    
    print(f"\n{'='*50}")
    print(f"光伏系统全流程分析程序 v1.2")
    print(f"站点: Lat {config.lat}, Lon {config.lon}")
    print(f"年份: {config.year}")
    print(f"容量: {config.capacity_mw} MW")
    print(f"{'='*50}\n")
    
    # 初始化 QC
    qc = QualityControl(output_dir=config.reports_dir)
    
    # 设置绘图样式
    Visualizer.set_style()

    try:
        # --- 2. 数据获取 ---
        print(">> 步骤 1/5: 获取气象数据...")
        dm = DataManager()
        weather_df = dm.get_weather_data(config.lat, config.lon, config.year, database=config.database)
        # 保存原始数据
        dm.save_raw_data(weather_df, config.lat, config.lon, config.year)
        print(f"   [成功] 获取 {len(weather_df)} 小时数据")
        
        # QC 检查
        qc.check_integrity(weather_df, "Weather Data")
        qc.detect_outliers(weather_df, ['ghi', 'dni', 'temp_air'])
        
        # --- 3. 光资源分析与绘图 ---
        print(">> 步骤 2/5: 分析光资源与绘图...")
        
        # [关键修复] 将绘图数据转换为本地时间 (Asia/Shanghai)
        weather_df_plot = weather_df.copy()
        if weather_df_plot.index.tz is None:
            weather_df_plot.index = weather_df_plot.index.tz_localize('UTC')
        weather_df_plot.index = weather_df_plot.index.tz_convert('Asia/Shanghai')
        
        analyzer = ResourceAnalyzer(weather_df)
        rad_stats = analyzer.analyze_radiation()
        
        # 静态绘图 (使用本地时间数据)
        print("   正在生成气象分析图表...")
        weather_fig_dir = Path(config.figures_dir) / "weather_plots"
        Visualizer.plot_weather_trends(weather_df_plot, weather_fig_dir)
        Visualizer.plot_weather_duration_curves(weather_df_plot, weather_fig_dir)
        Visualizer.plot_radiation_heatmaps(weather_df_plot, weather_fig_dir)
        Visualizer.plot_wind_analysis(weather_df_plot, weather_fig_dir)
        
        print(f"   - 年总辐射 (GHI): {rad_stats['yearly_irradiation']['ghi']:.2f} kWh/m²")
        print(f"   - 有效日照时数:   {rad_stats['sunshine_hours']} h")
        print(f"   - 年平均气温:     {rad_stats['avg_temp']:.2f} °C")
        
        # --- 4. 光伏模拟 ---
        print(">> 步骤 3/5: 模拟光伏出力...")
        sim = PVSimulator(
            config.lat, config.lon, 
            capacity_mw=config.capacity_mw,
            surface_tilt=config.surface_tilt,
            surface_azimuth=config.surface_azimuth,
            module_name=config.module_name,
            inverter_name=config.inverter_name
        )
        pv_results = sim.run(weather_df) # 使用原始 UTC 数据进行模拟
        print("   [成功] 光伏模拟完成")

        # --- 4.5 风电模拟 ---
        print(">> 步骤 3.5/5: 模拟风电出力...")
        wind_sim = WindSimulator(
            config.lat, config.lon, 
            capacity_mw=config.capacity_mw, 
            hub_height=config.hub_height, 
            turbine_type=config.turbine_type
        )
        wind_results = wind_sim.run(weather_df)
        print("   [成功] 风电模拟完成")

        # 整理结果
        output = pd.DataFrame({
            'GHI': weather_df['ghi'],
            'AC_Power_W': pv_results.ac,
            'Wind_Power_W': wind_results
        }, index=weather_df.index)
        
        # [关键修复] 输出数据也转为本地时间
        if output.index.tz is None:
            output.index = output.index.tz_localize('UTC')
        output.index = output.index.tz_convert('Asia/Shanghai')
        
        # --- 5. 数据处理与归一化 ---
        print(">> 步骤 4/5: 处理标幺值数据...")
        # 计算标幺值 (PU)
        output['PU'] = output['AC_Power_W'] / (config.capacity_mw * 1e6)
        output['Wind_PU'] = output['Wind_Power_W'] / (config.capacity_mw * 1e6)
        
        # 保存标幺值数据
        pu_save_dir = Path(config.output_dir)
        pu_save_dir.mkdir(parents=True, exist_ok=True)
        
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_filename = f"pv_wind_normalized_{date_str}.csv"
        combined_path = pu_save_dir / combined_filename
        output.to_csv(combined_path)
        print(f"   [保存] 标幺值数据已保存至 {combined_path}")
        
        # QC 检查 Output
        qc.check_integrity(output, "PV & Wind Output Data")
        qc.detect_outliers(output, ['AC_Power_W', 'Wind_Power_W'])
        
        # 统计指标 (光伏)
        pv_total_mwh = output['AC_Power_W'].sum() / 1e6
        pv_util_hours = pv_total_mwh / config.capacity_mw
        
        # 统计指标 (风电)
        wind_total_mwh = output['Wind_Power_W'].sum() / 1e6
        wind_util_hours = wind_total_mwh / config.capacity_mw
        
        print(f"   [光伏] 年总发电量: {pv_total_mwh:.2f} MWh")
        print(f"   [光伏] 等效利用小时: {pv_util_hours:.2f} h")
        print(f"   [光伏] 容量系数: {pv_util_hours/8760:.2%}")
        print(f"   [风电] 年总发电量: {wind_total_mwh:.2f} MWh")
        print(f"   [风电] 等效利用小时: {wind_util_hours:.2f} h")
        print(f"   [风电] 容量系数: {wind_util_hours/8760:.2%}")

        # --- 6. 生成图表报告 (并发执行) ---
        print(">> 步骤 5/5: 生成图表报告...")
        
        pv_fig_dir = Path(config.figures_dir) / "pv_plots"
        wind_fig_dir = Path(config.figures_dir) / "wind_plots"
        
        # 使用线程池并发生成独立图表
        with ThreadPoolExecutor(max_workers=2) as executor:
            # 提交光伏和风电图表生成任务
            future_pv = executor.submit(_plot_pv_charts, (output, pv_fig_dir))
            future_wind = executor.submit(_plot_wind_charts, (output, wind_fig_dir))
            
            # 等待完成
            future_pv.result()
            print("   光伏出力分析图表生成完成")
            future_wind.result()
            print("   风电出力分析图表生成完成")
        
        # 生成 QC 报告
        qc_report_filename = "qc_report.txt"
        qc.generate_report(qc_report_filename)
        print(f"   报告已保存至 {config.reports_dir}/{qc_report_filename}")

        print("\n[完成] 所有任务已完成！")
        print(f"       - 原始数据: {dm.raw_dir}/")
        print(f"       - 处理数据: {pu_save_dir}/")
        print(f"       - 静态图表: {config.figures_dir}/")
        print(f"       - 质量报告: {config.reports_dir}/{qc_report_filename}")

    except Exception as e:
        logger.critical(f"程序执行过程中发生严重错误: {e}", exc_info=True)
        print(f"\n[错误] 程序异常终止，详情请查看日志。")

if __name__ == "__main__":
    main()
