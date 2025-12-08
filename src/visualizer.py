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
        df['date'] = df.index.date
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
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
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
        ax.axvline(p90_val, color='green', linestyle='--', alpha=0.8)
        ax.axhline(p90_idx, color='green', linestyle='--', alpha=0.8)
        ax.text(p90_val + 0.02, p90_idx + 100, f'P90: {p90_val:.3f}', color='green', fontweight='bold')
        
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
