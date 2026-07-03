import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from pvlib.solarposition import get_solarposition

logger = logging.getLogger(__name__)

DAYTIME_GHI_THRESHOLD = 10  # W/m²，用于区分白天/夜间，避免夜间微小正值干扰


class QualityControl:
    """数据质量控制模块"""

    def __init__(self, output_dir: str = 'reports'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs = []

    def log(self, message: str):
        """记录日志，同时写入 logger"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self.logs.append(entry)
        logger.info(message)

    def check_integrity(self, df: pd.DataFrame, name: str = "Data") -> dict:
        """检查数据完整性"""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("check_integrity 需要 pd.DataFrame 输入")

        self.log(f"开始检查 {name} 完整性...")

        missing = df.isnull().sum()
        total_missing = int(missing.sum())

        time_gaps = 0
        duplicates = 0
        if isinstance(df.index, pd.DatetimeIndex):
            duplicates = int(df.index.duplicated().sum())
            expected_freq = pd.infer_freq(df.index)
            if expected_freq:
                full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=expected_freq, tz=df.index.tz)
                time_gaps = max(0, len(full_range) - len(df))
            else:
                self.log("无法推断时间频率，跳过连续性检查")
        else:
            self.log("输入数据没有时间索引，跳过连续性检查")

        self.log(f"缺失值总数: {total_missing}")
        if total_missing > 0:
            for col, count in missing[missing > 0].items():
                self.log(f"  - 列 '{col}': 缺失 {count} 个值")

        if time_gaps > 0:
            self.log(f"发现 {time_gaps} 个时间断点")
        if duplicates > 0:
            self.log(f"发现 {duplicates} 个重复时间戳")

        return {
            'missing': missing,
            'time_gaps': time_gaps,
            'duplicates': duplicates
        }

    def _daytime_mask(self, df: pd.DataFrame, col: str) -> pd.Series:
        """
        构建白天掩码。优先使用太阳高度角 > 0；
        若缺少经纬度信息，则退化为 ghi > DAYTIME_GHI_THRESHOLD。
        """
        if 'lat' in df.attrs and 'lon' in df.attrs and isinstance(df.index, pd.DatetimeIndex):
            try:
                solpos = get_solarposition(df.index, df.attrs['lat'], df.attrs['lon'])
                return solpos['apparent_elevation'] > 0
            except Exception:
                pass

        if 'ghi' in df.columns:
            return df['ghi'] > DAYTIME_GHI_THRESHOLD
        # 最后一道防线：当 ghi 也没有时，对正值数据做检测
        return df[col] > 0

    def detect_outliers(self, df: pd.DataFrame, columns: list, sigma: float = 3.0, physical_bounds: dict = None) -> dict:
        """
        检测异常值。
        - 对辐射数据（ghi, dni, dhi）仅白天时段参与 3-sigma 统计，避免夜间误报。
        - 对带物理边界的列（如功率输出），直接使用上下界进行判定。
        - 其他列使用全域 3-sigma。
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("detect_outliers 需要 pd.DataFrame 输入")

        physical_bounds = physical_bounds or {}
        self.log(f"开始检测异常值 (Sigma={sigma})...")
        outliers = {}

        for col in columns:
            if col not in df.columns:
                continue

            data = df[col]

            if col in physical_bounds:
                lower, upper = physical_bounds[col]
                mask = (data < lower) | (data > upper)
                count = int(mask.sum())
                if count > 0:
                    self.log(f"列 '{col}': 发现 {count} 个超出物理范围 [{lower:.2f}, {upper:.2f}] 的异常值")
                    outliers[col] = count
            elif col in ['ghi', 'dni', 'dhi']:
                daytime_mask = self._daytime_mask(df, col)
                daytime_data = data[daytime_mask]

                if len(daytime_data) > 0:
                    mean = daytime_data.mean()
                    std = daytime_data.std()
                    lower = mean - sigma * std
                    upper = mean + sigma * std

                    mask = ((data < lower) | (data > upper)) & daytime_mask
                    count = int(mask.sum())

                    if count > 0:
                        self.log(f"列 '{col}': 发现 {count} 个异常值 (白天范围: [{lower:.2f}, {upper:.2f}])")
                        outliers[col] = count
            else:
                mean = data.mean()
                std = data.std()
                lower = mean - sigma * std
                upper = mean + sigma * std
                mask = (data < lower) | (data > upper)
                count = int(mask.sum())

                if count > 0:
                    self.log(f"列 '{col}': 发现 {count} 个异常值 (范围: [{lower:.2f}, {upper:.2f}])")
                    outliers[col] = count

        if not outliers:
            self.log("未发现显著异常值")

        return outliers

    def generate_report(self, filename: str = "qc_report.txt") -> Path:
        """生成文本质量报告"""
        filepath = self.output_dir / filename
        self.log(f"生成质量控制报告: {filepath}")

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("质量控制报告 (Quality Control Report)\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            f.write("日志记录:\n")
            for entry in self.logs:
                f.write(entry + "\n")

        logger.info(f"报告已保存至 {filepath}")
        return filepath
