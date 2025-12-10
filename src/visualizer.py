import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import logging

class Visualizer:
    @staticmethod
    def set_style():
        """配置全局绘图样式"""
        # 先设置 Seaborn 主题
        sns.set_theme(style="whitegrid", font="Microsoft YaHei")
        
        # 再强制设置 Matplotlib 字体列表 (作为后备)
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
    @staticmethod
    def _save_figure(fig, save_path):
        """通用保存方法"""
        try:
            # 确保目录存在
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"图表已保存: {save_path}")
        except Exception as e:
            logging.error(f"图表保存失败: {e}")
        finally:
            plt.close(fig)

    @staticmethod
    def plot_weather_trends(df, save_dir):
        """
        1. 全年气象数据趋势图 (5子图)
        """
        save_path = Path(save_dir) / "weather_trends_yearly.png"
        
        # 准备数据 (0-8760)
        plot_data = df.reset_index(drop=True)
        x_axis = plot_data.index
        
        fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
        fig.suptitle("全年气象参数变化趋势", fontsize=16, y=0.92)
        
        params = [
            ('wind_speed', '风速 (m/s)', 'tab:green'),
            ('temp_air', '温度 (℃)', 'tab:red'),
            ('ghi', 'GHI (W/m²)', 'tab:orange'),
            ('dni', 'DNI (W/m²)', 'tab:blue'),
            ('dhi', 'DHI (W/m²)', 'tab:purple')
        ]
        
        for ax, (col, label, color) in zip(axes, params):
            if col in plot_data.columns:
                ax.plot(x_axis, plot_data[col], color=color, linewidth=0.5)
                ax.set_ylabel(label)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'Data Not Available', ha='center')
                
        axes[-1].set_xlabel("时间（小时）")
        axes[-1].set_xlim(0, 8760)
        
        Visualizer._save_figure(fig, save_path)

    @staticmethod
    def plot_wind_analysis(weather_df, save_dir):
        """
        绘制风资源分析图表：
        1. 风速频率分布 + Weibull 拟合
        2. 风速月变化箱线图
        3. 风速日变化曲线
        """
        from scipy.stats import weibull_min

        # 1. 风速频率分布
        save_path_dist = Path(save_dir) / "wind_speed_distribution.png"
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        
        data = weather_df['wind_speed'].dropna()
        
        # 直方图
        sns.histplot(data, stat="density", element="step", fill=True, color='skyblue', label='观测数据', ax=ax1)
        
        # Weibull 拟合
        if len(data) > 0:
            try:
                params = weibull_min.fit(data, floc=0)
                x = np.linspace(0, data.max() + 5, 100)
                ax1.plot(x, weibull_min.pdf(x, *params), 'r-', lw=2, label=f'Weibull 拟合 (k={params[0]:.2f}, c={params[2]:.2f})')
            except Exception as e:
                logging.warning(f"Weibull fitting failed: {e}")
            
        ax1.set_xlabel("风速 (m/s)")
        ax1.set_ylabel("概率密度")
        ax1.set_title("风速频率分布与 Weibull 拟合")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        Visualizer._save_figure(fig1, save_path_dist)

        # 2. 月变化 & 日变化 (合并在一个图中)
        save_path_prof = Path(save_dir) / "wind_speed_profiles.png"
        fig2, (ax2, ax3) = plt.subplots(2, 1, figsize=(12, 12))
        
        # 月变化 (箱线图)
        # 确保有 month 列 (如果不修改原始 df，用 copy)
        df_plot = weather_df.copy()
        df_plot['month'] = df_plot.index.month
        try:
            sns.boxplot(x='month', y='wind_speed', data=df_plot, ax=ax2, palette="Blues", hue='month', legend=False)
        except TypeError:
            # 旧版本 seaborn 兼容
            sns.boxplot(x='month', y='wind_speed', data=df_plot, ax=ax2, palette="Blues")
        ax2.set_xlabel("月份")
        ax2.set_ylabel("风速 (m/s)")
        ax2.set_title("风速月变化特征")
        ax2.grid(True, axis='y', alpha=0.3)
        
        # 日变化 (平均曲线 + 标准差)
        df_plot['hour'] = df_plot.index.hour
        # seaborn >= 0.12 uses errorbar='sd', older uses ci='sd'
        try:
            sns.lineplot(x='hour', y='wind_speed', data=df_plot, ax=ax3, errorbar='sd', color='tab:blue')
        except TypeError:
            sns.lineplot(x='hour', y='wind_speed', data=df_plot, ax=ax3, ci='sd', color='tab:blue')
            
        ax3.set_xlabel("小时 (0-23)")
        ax3.set_ylabel("风速 (m/s)")
        ax3.set_title("风速日变化特征 (平均值 ± 标准差)")
        ax3.set_xlim(0, 23)
        ax3.set_xticks(range(0, 24, 2))
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        Visualizer._save_figure(fig2, save_path_prof)

    @staticmethod
    def plot_wind_power_analysis(output_df, save_dir):
        """
        绘制风电出力分析图表:
        1. 出力热力图
        2. 出力 Duration Curve
        """
        # 1. 热力图
        save_path_hm = Path(save_dir) / "wind_heatmap.png"
        matrix = Visualizer._prepare_heatmap_data(output_df['Wind_PU'])
        
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        im = ax1.imshow(matrix, aspect='auto', cmap='viridis', origin='lower', extent=[0, 365, 0, 24], vmin=0, vmax=1)
        ax1.set_title("风电出力标幺值年分布热力图", fontsize=14)
        ax1.set_xlabel("天数 (1-365)")
        ax1.set_ylabel("小时 (0-23)")
        
        month_starts = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax1.set_xticks([x + 15 for x in month_starts])
        ax1.set_xticklabels(month_names)
        
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label("标幺值")
        Visualizer._save_figure(fig1, save_path_hm)
        
        # 2. Duration Curve
        save_path_dc = Path(save_dir) / "wind_duration_curve.png"
        sorted_pu = output_df['Wind_PU'].sort_values(ascending=False).reset_index(drop=True)
        hours = np.arange(len(sorted_pu))
        
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        ax2.plot(sorted_pu, hours, color='tab:green', linewidth=2)
        ax2.set_xlabel("风电出力标幺值 (0-1)")
        ax2.set_ylabel("累计小时数 (0-8760)")
        ax2.set_title("风电出力持续曲线 (Duration Curve)", fontsize=14)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 8760)
        ax2.grid(True, alpha=0.3)
        
        # P50/P90
        p50_idx = int(len(sorted_pu) * 0.5)
        p90_idx = int(len(sorted_pu) * 0.9)
        p50_val = sorted_pu.iloc[p50_idx]
        p90_val = sorted_pu.iloc[p90_idx]
        
        ax2.axvline(p50_val, color='orange', linestyle='--', alpha=0.8)
        ax2.axhline(p50_idx, color='orange', linestyle='--', alpha=0.8)
        ax2.text(p50_val + 0.02, p50_idx + 100, f'P50: {p50_val:.3f}', color='orange', fontweight='bold')
        
        ax2.axvline(p90_val, color='red', linestyle='--', alpha=0.8)
        ax2.axhline(p90_idx, color='red', linestyle='--', alpha=0.8)
        ax2.text(p90_val + 0.02, p90_idx + 100, f'P90: {p90_val:.3f}', color='red', fontweight='bold')
        
        Visualizer._save_figure(fig2, save_path_dc)

    @staticmethod
    def plot_weather_duration_curves(df, save_dir):
        """
        3.1 气象参数 Duration Curve (频数分布)
        """
        save_path = Path(save_dir) / "weather_duration_curves.png"
        
        fig, axes = plt.subplots(2, 3, figsize=(12, 8)) # 5个图，用2x3网格
        fig.suptitle("气象参数频数分布图", fontsize=16)
        axes = axes.flatten()
        
        params = [
            ('dni', 'DNI (W/m²)', 'tab:blue'),
            ('ghi', 'GHI (W/m²)', 'tab:orange'),
            ('dhi', 'DHI (W/m²)', 'tab:purple'),
            ('temp_air', '温度 (℃)', 'tab:red'),
            ('wind_speed', '风速 (m/s)', 'tab:green')
        ]
        
        for i, (col, label, color) in enumerate(params):
            ax = axes[i]
            if col in df.columns:
                # 绘制直方图 (频数)
                sns.histplot(df[col], element="step", fill=False, ax=ax, color=color)
                ax.set_xlabel(label)
                ax.set_ylabel("出现频数（小时）")
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center')
        
        # 移除第6个空图
        fig.delaxes(axes[5])
        plt.tight_layout()
        Visualizer._save_figure(fig, save_path)

    @staticmethod
    def _prepare_heatmap_data(series):
        """
        将时间序列转换为 (24, 365) 矩阵，确保时间对齐
        """
        # 1. 确保是 DataFrame
        if isinstance(series, pd.Series):
            df = series.to_frame(name='value')
        else:
            df = series.copy()
            df.columns = ['value']
            
        # 2. 提取小时和日期 (使用 Local Time)
        # 假设 index 已经是 datetime 格式，如果有时区，直接用
        df['hour'] = df.index.hour
        # df['date'] = df.index.date
        
        # 处理闰年移除2月29日后的 day_of_year 偏移问题
        is_leap = df.index.is_leap_year.any() if len(df) > 0 else False
        if is_leap:
            doy = df.index.dayofyear
            has_leap_day = ((df.index.month == 2) & (df.index.day == 29)).any()
            if not has_leap_day:
                # 如果是闰年且移除了2月29日，将3月1日及以后的 day_of_year 减1，填补第60天的空缺
                df['day_of_year'] = np.where(doy > 60, doy - 1, doy)
            else:
                df['day_of_year'] = doy
        else:
            df['day_of_year'] = df.index.dayofyear
        
        # 3. Pivot Table
        # 行: Hour (0-23)
        # 列: Date (Day of Year 1-365/366)
        # values: value
        pivot = df.pivot_table(index='hour', columns='day_of_year', values='value')
        
        # 4. 补全缺失的小时或天数
        # 确保 index 是 0-23
        pivot = pivot.reindex(index=range(24), fill_value=0)
        
        # 确保 columns 是 1-365 (忽略闰年多出的一天，或者保留)
        # 为了热力图展示一致性，通常截断为 365 或补全
        expected_days = range(1, 366)
        pivot = pivot.reindex(columns=expected_days, fill_value=0)
        
        # 填充可能存在的 NaN (例如某天缺数据)
        pivot = pivot.fillna(0)
        
        # 5. 转换为 numpy 矩阵
        # pivot.values 形状为 (24, 365)
        return pivot.values

    @staticmethod
    def plot_pv_heatmap(output_df, save_dir):
        """
        3.2 出力标幺值热力图
        """
        save_path = Path(save_dir) / "pv_heatmap.png"
        
        matrix = Visualizer._prepare_heatmap_data(output_df['PU'])
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 绘制热力图
        im = ax.imshow(matrix, aspect='auto', cmap='viridis', origin='lower',
                       extent=[0, 365, 0, 24], vmin=0, vmax=1)
        
        ax.set_title("光伏出力标幺值年分布热力图", fontsize=14)
        ax.set_xlabel("天数 (1-365)")
        ax.set_ylabel("小时 (0-23)")
        
        # 设置月份刻度
        month_starts = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticks([x + 15 for x in month_starts]) # 居中
        ax.set_xticklabels(month_names)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("标幺值")
        
        Visualizer._save_figure(fig, save_path)

    @staticmethod
    def plot_pv_duration_curve(output_df, save_dir):
        """
        3.3 出力 Duration Curve
        """
        save_path = Path(save_dir) / "pv_duration_curve.png"
        
        # 排序 (从大到小)
        sorted_pu = output_df['PU'].sort_values(ascending=False).reset_index(drop=True)
        hours = np.arange(len(sorted_pu))
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 绘制曲线 (横轴PU，纵轴累计小时) -> 用户要求: 横轴标幺值，纵轴累计小时数
        # 通常 Duration Curve 是 X=Hours, Y=Power. 但用户要求 "横轴：光伏出力标幺值", "纵轴：累计小时数"
        # 这意味着：对于某个出力水平 P，有多少小时出力 >= P (累积分布函数翻转)
        # 或者仅仅是散点图/直方图的累积形式
        # 按照用户描述 "横轴：标幺值（0-1），纵轴：累计小时数（0-8760）"
        # 这实际上是累积分布函数 (CDF) 的变体
        
        # 我们用 sort_values 得到的是 Y=PU, X=Rank(Hours).
        # 用户要求 X=PU, Y=Hours. 我们可以交换 XY 轴绘制
        
        ax.plot(sorted_pu, hours, color='tab:blue', linewidth=2)
        ax.set_xlabel("光伏出力标幺值 (0-1)")
        ax.set_ylabel("累计小时数 (0-8760)")
        ax.set_title("光伏出力持续曲线 (Duration Curve)", fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 8760)
        ax.grid(True, alpha=0.3)
        
        # P50, P90
        # P50: 概率 50% 的值 (即 8760/2 = 4380 小时对应的 PU)
        # P90: 概率 90% 的值 (即 8760*0.9 = 7884 小时对应的 PU)
        p50_idx = int(len(sorted_pu) * 0.5)
        p90_idx = int(len(sorted_pu) * 0.9)
        
        p50_val = sorted_pu.iloc[p50_idx]
        p90_val = sorted_pu.iloc[p90_idx]
        
        # 标记线
        # P50
        ax.axvline(p50_val, color='orange', linestyle='--', alpha=0.8)
        ax.axhline(p50_idx, color='orange', linestyle='--', alpha=0.8)
        ax.text(p50_val + 0.02, p50_idx + 100, f'P50: {p50_val:.3f}', color='orange', fontweight='bold')
        
        # P90
        ax.axvline(p90_val, color='red', linestyle='--', alpha=0.8)
        ax.axhline(p90_idx, color='red', linestyle='--', alpha=0.8)
        ax.text(p90_val + 0.02, p90_idx + 100, f'P90: {p90_val:.3f}', color='red', fontweight='bold')
        
        Visualizer._save_figure(fig, save_path)

    @staticmethod
    def plot_radiation_heatmaps(weather_df, save_dir):
        """
        3.4 DNI/GHI 热力图
        """
        save_path = Path(save_dir) / "radiation_heatmaps.png"
        
        dni_mat = Visualizer._prepare_heatmap_data(weather_df['dni'])
        ghi_mat = Visualizer._prepare_heatmap_data(weather_df['ghi'])
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # DNI
        im1 = axes[0].imshow(dni_mat, aspect='auto', cmap='plasma', origin='lower', extent=[0, 365, 0, 24])
        axes[0].set_title("DNI 年分布热力图")
        axes[0].set_ylabel("小时 (0-23)")
        cbar1 = plt.colorbar(im1, ax=axes[0])
        cbar1.set_label("DNI (W/m²)")
        
        # GHI
        im2 = axes[1].imshow(ghi_mat, aspect='auto', cmap='plasma', origin='lower', extent=[0, 365, 0, 24])
        axes[1].set_title("GHI 年分布热力图")
        axes[1].set_ylabel("小时 (0-23)")
        axes[1].set_xlabel("天数 (1-365)")
        cbar2 = plt.colorbar(im2, ax=axes[1])
        cbar2.set_label("GHI (W/m²)")
        
        # Month ticks
        month_starts = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        axes[1].set_xticks([x + 15 for x in month_starts])
        axes[1].set_xticklabels(month_names)
        
        Visualizer._save_figure(fig, save_path)
