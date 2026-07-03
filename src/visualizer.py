import matplotlib
matplotlib.use('Agg')  # 非交互式后端，线程安全，适合自动化/测试

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

MONTH_STARTS = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


class Visualizer:
    @staticmethod
    def set_style():
        """配置全局绘图样式"""
        sns.set_theme(style="whitegrid", font="Microsoft YaHei")
        plt.rcParams['font.sans-serif'] = [
            'Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS', 'DejaVu Sans'
        ]
        plt.rcParams['axes.unicode_minus'] = False

    @staticmethod
    def _save_figure(fig, save_path: Path):
        """通用保存方法"""
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"图表已保存: {save_path}")
        except Exception as e:
            logger.error(f"图表保存失败: {e}")
            raise
        finally:
            plt.close(fig)

    @staticmethod
    def plot_weather_trends(df: pd.DataFrame, save_dir: Path):
        """全年气象数据趋势图"""
        save_path = Path(save_dir) / "weather_trends_yearly.png"

        plot_data = df.copy()
        numeric_cols = ['wind_speed', 'temp_air', 'ghi', 'dni', 'dhi']
        for col in numeric_cols:
            if col in plot_data.columns:
                plot_data[col] = plot_data[col].rolling(window=24, center=True, min_periods=1).mean()

        plot_data = plot_data.reset_index(drop=True)
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
                ax.plot(x_axis, plot_data[col], color=color, linewidth=0.8)
                ax.set_ylabel(label)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'Data Not Available', ha='center', transform=ax.transAxes)

        axes[-1].set_xlabel("时间（小时）")
        axes[-1].set_xlim(0, 8760)

        Visualizer._save_figure(fig, save_path)

    @staticmethod
    def plot_wind_analysis(weather_df: pd.DataFrame, save_dir: Path):
        """风资源分析图表"""
        from scipy.stats import weibull_min

        save_path_dist = Path(save_dir) / "wind_speed_distribution.png"
        fig1, ax1 = plt.subplots(figsize=(12, 8))

        data = weather_df['wind_speed'].dropna()
        sns.histplot(data, stat="density", element="step", fill=True, color='skyblue', label='观测数据', ax=ax1)

        if len(data) > 0:
            try:
                params = weibull_min.fit(data, floc=0)
                x = np.linspace(0, data.max() + 5, 100)
                ax1.plot(x, weibull_min.pdf(x, *params), 'r-', lw=2,
                         label=f'Weibull 拟合 (k={params[0]:.2f}, c={params[2]:.2f})')
            except Exception as e:
                logger.warning(f"Weibull fitting failed: {e}")

        ax1.set_xlabel("风速 (m/s)")
        ax1.set_ylabel("概率密度")
        ax1.set_title("风速频率分布与 Weibull 拟合")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        Visualizer._save_figure(fig1, save_path_dist)

        save_path_prof = Path(save_dir) / "wind_speed_profiles.png"
        fig2, (ax2, ax3) = plt.subplots(2, 1, figsize=(12, 12))

        df_plot = weather_df.copy()
        df_plot['month'] = df_plot.index.month
        sns.boxplot(x='month', y='wind_speed', data=df_plot, ax=ax2, palette="Blues", hue='month', legend=False)
        ax2.set_xlabel("月份")
        ax2.set_ylabel("风速 (m/s)")
        ax2.set_title("风速月变化特征")
        ax2.grid(True, axis='y', alpha=0.3)

        df_plot['hour'] = df_plot.index.hour
        sns.lineplot(x='hour', y='wind_speed', data=df_plot, ax=ax3, errorbar='sd', color='tab:blue')
        ax3.set_xlabel("小时 (0-23)")
        ax3.set_ylabel("风速 (m/s)")
        ax3.set_title("风速日变化特征 (平均值 ± 标准差)")
        ax3.set_xlim(0, 23)
        ax3.set_xticks(range(0, 24, 2))
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        Visualizer._save_figure(fig2, save_path_prof)

    @staticmethod
    def plot_wind_power_analysis(output_df: pd.DataFrame, save_dir: Path):
        """风电出力分析图表"""
        save_path_hm = Path(save_dir) / "wind_heatmap.png"
        matrix = Visualizer._prepare_heatmap_data(output_df['Wind_PU'])

        fig1, ax1 = plt.subplots(figsize=(12, 8))
        im = ax1.imshow(matrix, aspect='auto', cmap='viridis', origin='lower',
                        extent=[0, 365, 0, 24], vmin=0, vmax=1)
        ax1.set_title("风电出力标幺值年分布热力图", fontsize=14)
        ax1.set_xlabel("天数 (1-365)")
        ax1.set_ylabel("小时 (0-23)")
        ax1.set_xticks([x + 15 for x in MONTH_STARTS])
        ax1.set_xticklabels(MONTH_NAMES)
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label("标幺值")
        Visualizer._save_figure(fig1, save_path_hm)

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

        p50_idx = int(len(sorted_pu) * 0.5)
        p90_idx = int(len(sorted_pu) * 0.9)
        p50_val = sorted_pu.iloc[p50_idx]
        p90_val = sorted_pu.iloc[p90_idx]

        ax2.axvline(p50_val, color='orange', linestyle='--', alpha=0.8)
        ax2.axhline(p50_idx, color='orange', linestyle='--', alpha=0.8)
        ax2.text(p50_val + 0.02, p50_idx + 100, f'P50: {p50_val:.3f}',
                 color='orange', fontweight='bold')

        ax2.axvline(p90_val, color='red', linestyle='--', alpha=0.8)
        ax2.axhline(p90_idx, color='red', linestyle='--', alpha=0.8)
        ax2.text(p90_val + 0.02, p90_idx + 100, f'P90: {p90_val:.3f}',
                 color='red', fontweight='bold')

        Visualizer._save_figure(fig2, save_path_dc)

    @staticmethod
    def plot_weather_histograms(df: pd.DataFrame, save_dir: Path):
        """气象参数频数分布图"""
        save_path = Path(save_dir) / "weather_histograms.png"

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
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
                sns.histplot(df[col], element="step", fill=True, ax=ax, color=color)
                ax.set_xlabel(label)
                ax.set_ylabel("出现频数（小时）")
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center')

        fig.delaxes(axes[5])
        plt.tight_layout()
        Visualizer._save_figure(fig, save_path)

    @staticmethod
    def _prepare_heatmap_data(series: pd.Series) -> np.ndarray:
        """将时间序列转换为 (24, 365) 矩阵"""
        if isinstance(series, pd.Series):
            df = series.to_frame(name='value')
        else:
            df = series.copy()
            df.columns = ['value']

        df['hour'] = df.index.hour

        is_leap = df.index.is_leap_year.any() if len(df) > 0 else False
        doy = df.index.dayofyear
        has_leap_day = ((df.index.month == 2) & (df.index.day == 29)).any()

        if is_leap and not has_leap_day:
            df['day_of_year'] = np.where(doy > 60, doy - 1, doy)
        else:
            df['day_of_year'] = doy

        pivot = df.pivot_table(index='hour', columns='day_of_year', values='value')
        pivot = pivot.reindex(index=range(24), fill_value=0)
        pivot = pivot.reindex(columns=range(1, 366), fill_value=0)
        pivot = pivot.fillna(0)
        return pivot.values

    @staticmethod
    def plot_pv_heatmap(output_df: pd.DataFrame, save_dir: Path):
        """光伏出力标幺值热力图"""
        save_path = Path(save_dir) / "pv_heatmap.png"
        matrix = Visualizer._prepare_heatmap_data(output_df['PU'])

        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(matrix, aspect='auto', cmap='viridis', origin='lower',
                       extent=[0, 365, 0, 24], vmin=0, vmax=1)
        ax.set_title("光伏出力标幺值年分布热力图", fontsize=14)
        ax.set_xlabel("天数 (1-365)")
        ax.set_ylabel("小时 (0-23)")
        ax.set_xticks([x + 15 for x in MONTH_STARTS])
        ax.set_xticklabels(MONTH_NAMES)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("标幺值")
        Visualizer._save_figure(fig, save_path)

    @staticmethod
    def plot_pv_duration_curve(output_df: pd.DataFrame, save_dir: Path):
        """光伏出力 Duration Curve"""
        save_path = Path(save_dir) / "pv_duration_curve.png"
        sorted_pu = output_df['PU'].sort_values(ascending=False).reset_index(drop=True)
        hours = np.arange(len(sorted_pu))

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(sorted_pu, hours, color='tab:blue', linewidth=2)
        ax.set_xlabel("光伏出力标幺值 (0-1)")
        ax.set_ylabel("累计小时数 (0-8760)")
        ax.set_title("光伏出力持续曲线 (Duration Curve)", fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 8760)
        ax.grid(True, alpha=0.3)

        p50_idx = int(len(sorted_pu) * 0.5)
        p90_idx = int(len(sorted_pu) * 0.9)
        p50_val = sorted_pu.iloc[p50_idx]
        p90_val = sorted_pu.iloc[p90_idx]

        ax.axvline(p50_val, color='orange', linestyle='--', alpha=0.8)
        ax.axhline(p50_idx, color='orange', linestyle='--', alpha=0.8)
        ax.text(p50_val + 0.02, p50_idx + 100, f'P50: {p50_val:.3f}',
                color='orange', fontweight='bold')

        ax.axvline(p90_val, color='red', linestyle='--', alpha=0.8)
        ax.axhline(p90_idx, color='red', linestyle='--', alpha=0.8)
        ax.text(p90_val + 0.02, p90_idx + 100, f'P90: {p90_val:.3f}',
                color='red', fontweight='bold')

        Visualizer._save_figure(fig, save_path)

    @staticmethod
    def plot_radiation_heatmaps(weather_df: pd.DataFrame, save_dir: Path):
        """DNI/GHI 热力图"""
        save_path = Path(save_dir) / "radiation_heatmaps.png"
        dni_mat = Visualizer._prepare_heatmap_data(weather_df['dni'])
        ghi_mat = Visualizer._prepare_heatmap_data(weather_df['ghi'])

        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        im1 = axes[0].imshow(dni_mat, aspect='auto', cmap='plasma', origin='lower', extent=[0, 365, 0, 24])
        axes[0].set_title("DNI 年分布热力图")
        axes[0].set_ylabel("小时 (0-23)")
        cbar1 = plt.colorbar(im1, ax=axes[0])
        cbar1.set_label("DNI (W/m²)")

        im2 = axes[1].imshow(ghi_mat, aspect='auto', cmap='plasma', origin='lower', extent=[0, 365, 0, 24])
        axes[1].set_title("GHI 年分布热力图")
        axes[1].set_ylabel("小时 (0-23)")
        axes[1].set_xlabel("天数 (1-365)")
        cbar2 = plt.colorbar(im2, ax=axes[1])
        cbar2.set_label("GHI (W/m²)")

        axes[1].set_xticks([x + 15 for x in MONTH_STARTS])
        axes[1].set_xticklabels(MONTH_NAMES)

        Visualizer._save_figure(fig, save_path)
