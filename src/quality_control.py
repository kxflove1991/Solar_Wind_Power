import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime


class QualityControl:
    def __init__(self, output_dir='reports'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logs = []

    def log(self, message):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self.logs.append(entry)
        logging.info(message)

    def check_integrity(self, df, name="Data"):
        """
        检查数据完整性
        :param df: DataFrame
        :param name: 数据集名称
        :return: 检查结果字典
        """
        self.log(f"开始检查 {name} 完整性...")
        
        # 1. 缺失值检查
        missing = df.isnull().sum()
        total_missing = missing.sum()
        
        # 2. 时间连续性检查 (假设是时间索引)
        time_gaps = 0
        if isinstance(df.index, pd.DatetimeIndex):
            expected_freq = pd.infer_freq(df.index)
            if expected_freq:
                full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=expected_freq)
                time_gaps = len(full_range) - len(df)
            else:
                self.log("无法推断时间频率，跳过连续性检查")
        
        # 3. 记录结果
        self.log(f"缺失值总数: {total_missing}")
        if total_missing > 0:
            for col, count in missing[missing > 0].items():
                self.log(f"  - 列 '{col}': 缺失 {count} 个值")
        
        if time_gaps > 0:
            self.log(f"发现 {time_gaps} 个时间断点")
        
        return {
            'missing': missing,
            'time_gaps': time_gaps
        }

    def detect_outliers(self, df, columns, sigma=3):
        """
        基于 3-sigma 原则检测异常值
        """
        self.log(f"开始检测异常值 (Sigma={sigma})...")
        outliers = {}
        
        for col in columns:
            if col not in df.columns:
                continue
                
            data = df[col]
            mean = data.mean()
            std = data.std()
            
            # 定义阈值
            lower = mean - sigma * std
            upper = mean + sigma * std
            
            # 查找异常
            mask = (data < lower) | (data > upper)
            count = mask.sum()
            
            if count > 0:
                self.log(f"列 '{col}': 发现 {count} 个异常值 (范围: [{lower:.2f}, {upper:.2f}])")
                outliers[col] = count
                # 标记异常值 (可选：添加一列标记)
                # df[f'{col}_flag'] = mask
        
        if not outliers:
            self.log("未发现显著异常值")
            
        return outliers

    def generate_report(self, filename="qc_report.txt"):
        """
        生成文本质量报告 (替代 PDF)
        """
        filepath = self.output_dir / filename
        self.log(f"生成质量控制报告: {filepath}")
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("="*50 + "\n")
                f.write("质量控制报告 (Quality Control Report)\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*50 + "\n\n")
                
                f.write("日志记录:\n")
                for entry in self.logs:
                    f.write(entry + "\n")
                    
            logging.info(f"报告已保存至 {filepath}")
        except Exception as e:
            logging.error(f"保存报告失败: {e}")


