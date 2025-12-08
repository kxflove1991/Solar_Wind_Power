import pandas as pd
from src.data_manager import DataManager
from src.analyzer import ResourceAnalyzer
from src.simulator import PVSimulator
from src.visualizer import Visualizer
from src.quality_control import QualityControl
import os
import logging
from datetime import datetime
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # --- 1. 配置参数 ---
    # 示例站点：内蒙古阿拉善右旗
    LAT, LON = 39.0, 102.0
    YEAR = 2020
    CAPACITY_MW = 100.0 # 典型大基地项目规模
    
    print(f"\n{'='*50}")
    print(f"🌞 光伏系统全流程分析程序 v1.1")
    print(f"站点: Lat {LAT}, Lon {LON}")
    print(f"年份: {YEAR}")
    print(f"容量: {CAPACITY_MW} MW")
    print(f"{'='*50}\n")
    
    # 初始化 QC
    qc = QualityControl()
    
    # 设置绘图样式
    Visualizer.set_style()

    # --- 2. 数据获取 ---
    print(">> 步骤 1/5: 获取气象数据...")
    dm = DataManager()
    try:
        weather_df = dm.get_weather_data(LAT, LON, YEAR)
        # 保存原始数据 (Requirement 1)
        dm.save_raw_data(weather_df, LAT, LON)
        print(f"   [成功] 获取 {len(weather_df)} 小时数据")
        
        # QC 检查 (Requirement 5)
        qc.check_integrity(weather_df, "Weather Data")
        qc.detect_outliers(weather_df, ['ghi', 'dni', 'temp_air'])
        
    except Exception as e:
        print(f"   [失败] {e}")
        return
    
    # --- 3. 光资源分析与绘图 ---
    print(">> 步骤 2/5: 分析光资源与绘图...")
    
    # [关键修复] 将绘图数据转换为本地时间 (Asia/Shanghai)
    # 注意：仅用于绘图，模拟仍使用 UTC 数据以确保 pvlib 计算准确
    weather_df_plot = weather_df.copy()
    if weather_df_plot.index.tz is None:
        weather_df_plot.index = weather_df_plot.index.tz_localize('UTC')
    weather_df_plot.index = weather_df_plot.index.tz_convert('Asia/Shanghai')
    
    analyzer = ResourceAnalyzer(weather_df)
    rad_stats = analyzer.analyze_radiation()
    
    # 静态绘图 (使用本地时间数据)
    print("   正在生成气象分析图表...")
    weather_fig_dir = "figures/weather_plots"
    Visualizer.plot_weather_trends(weather_df_plot, weather_fig_dir)
    Visualizer.plot_weather_duration_curves(weather_df_plot, weather_fig_dir)
    Visualizer.plot_radiation_heatmaps(weather_df_plot, weather_fig_dir)
    
    print(f"   - 年总辐射 (GHI): {rad_stats['yearly_irradiation']['ghi']:.2f} kWh/m²")
    print(f"   - 有效日照时数:   {rad_stats['sunshine_hours']} h")
    print(f"   - 年平均气温:     {rad_stats['avg_temp']:.2f} °C")
    
    # --- 4. 光伏模拟 ---
    print(">> 步骤 3/5: 模拟光伏出力...")
    # 可选：指定组件和逆变器型号
    sim = PVSimulator(LAT, LON, capacity_mw=CAPACITY_MW)
    try:
        results = sim.run(weather_df) # 使用原始 UTC 数据进行模拟
        print("   [成功] 模拟完成")
    except Exception as e:
        print(f"   [失败] {e}")
        return
    
    # 整理结果
    output = pd.DataFrame({
        'GHI': weather_df['ghi'],
        'AC_Power_W': results.ac
    }, index=weather_df.index)
    
    # [关键修复] 输出数据也转为本地时间，以便绘图和分析符合直觉
    if output.index.tz is None:
        output.index = output.index.tz_localize('UTC')
    output.index = output.index.tz_convert('Asia/Shanghai')
    
    # --- 5. 数据处理与归一化 ---
    print(">> 步骤 4/5: 处理标幺值数据...")
    # 计算标幺值 (PU)
    output['PU'] = output['AC_Power_W'] / (CAPACITY_MW * 1e6)
    
    # 保存归一化数据 (Requirement 3)
    pu_save_dir = Path("data/processed/pv_normalized")
    pu_save_dir.mkdir(parents=True, exist_ok=True)
    
    date_str = datetime.now().strftime("%Y%m%d")
    pu_filename = f"pv_normalized_{date_str}.csv"
    output.to_csv(pu_save_dir / pu_filename)
    print(f"   [保存] 标幺值数据已保存至 {pu_save_dir}/{pu_filename}")

    # QC 检查 Output
    qc.check_integrity(output, "PV Output Data")
    qc.detect_outliers(output, ['AC_Power_W'])

    # 统计指标
    total_gen_mwh = output['AC_Power_W'].sum() / 1e6
    util_hours = total_gen_mwh / CAPACITY_MW
    
    print(f"   - 年总发电量: {total_gen_mwh:.2f} MWh")
    print(f"   - 等效利用小时: {util_hours:.2f} h")
    print(f"   - 容量系数: {util_hours/8760:.2%}")

    # --- 6. 可视化报告 ---
    print(">> 步骤 5/5: 生成图表报告...")
    
    # 静态绘图 (Requirement 4)
    print("   正在生成光伏出力分析图表...")
    pv_fig_dir = "figures/pv_plots"
    Visualizer.plot_pv_heatmap(output, pv_fig_dir)
    Visualizer.plot_pv_duration_curve(output, pv_fig_dir)
    
    # 生成 QC 报告 (Requirement 5)
    qc.generate_report("qc_report.txt")
    
    # 交互式 HTML 报告 (移除旧的 Plotly 调用，如有需要可后续添加)
    # report_dir = "reports"
    # os.makedirs(report_dir, exist_ok=True)
    # 
    # # 5.1 GHI 热力图 (已由 plot_radiation_heatmaps 替代)
    # 
    # # 5.2 月度发电量 (已移除)
    # 
    # # 5.3 典型日曲线 (已移除)
    
    print(f"\n[完成] 所有任务已完成！")
    print(f"       - 原始数据: ./data/raw/weather/")
    print(f"       - 处理数据: ./data/processed/pv_normalized/")
    print(f"       - 静态图表: ./figures/")
    print(f"       - 质量报告: ./reports/qc_report.txt")

if __name__ == "__main__":
    main()
