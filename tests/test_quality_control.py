import pytest
import pandas as pd
import numpy as np
from src.quality_control import QualityControl


@pytest.fixture
def sample_df():
    idx = pd.date_range('2020-01-01', periods=100, freq='h', tz='UTC')
    df = pd.DataFrame({
        'ghi': np.concatenate([np.zeros(50), np.linspace(0, 800, 50)]),
        'dni': np.concatenate([np.zeros(50), np.linspace(0, 700, 50)]),
        'temp_air': np.random.normal(20, 5, 100),
        'wind_speed': np.random.normal(5, 2, 100),
    }, index=idx)
    df.attrs['lat'] = 39.0
    df.attrs['lon'] = 102.0
    return df


class TestQualityControl:
    def test_check_integrity(self, sample_df, tmp_path):
        qc = QualityControl(output_dir=str(tmp_path))
        result = qc.check_integrity(sample_df, "Test Data")

        assert result['time_gaps'] == 0
        assert result['missing'].sum() == 0

    def test_detect_outliers_no_outliers(self, sample_df, tmp_path):
        qc = QualityControl(output_dir=str(tmp_path))
        outliers = qc.detect_outliers(sample_df, ['ghi', 'dni', 'temp_air'])
        # 数据线性分布，正常情况不应有 3-sigma 异常
        assert 'ghi' not in outliers
        assert 'dni' not in outliers

    def test_detect_outliers_with_outliers(self, sample_df, tmp_path):
        df = sample_df.copy()
        df.loc[df.index[75], 'ghi'] = 5000  # 白天异常高值
        qc = QualityControl(output_dir=str(tmp_path))
        outliers = qc.detect_outliers(df, ['ghi'])
        assert outliers.get('ghi', 0) >= 1

    def test_detect_outliers_night_not_flagged(self, sample_df, tmp_path):
        df = sample_df.copy()
        # 选择明确的夜间时刻（凌晨 2 点）设置异常高值
        df.loc[df.index[2], 'ghi'] = 1000
        qc = QualityControl(output_dir=str(tmp_path))
        outliers = qc.detect_outliers(df, ['ghi'])
        # 夜间时段不应被计入白天异常
        assert outliers.get('ghi', 0) == 0

    def test_generate_report(self, sample_df, tmp_path):
        qc = QualityControl(output_dir=str(tmp_path))
        qc.check_integrity(sample_df, "Test Data")
        path = qc.generate_report("test_qc_report.txt")
        assert path.exists()
        content = path.read_text(encoding='utf-8')
        assert "Test Data" in content

    def test_invalid_input_type(self, tmp_path):
        qc = QualityControl(output_dir=str(tmp_path))
        with pytest.raises(TypeError):
            qc.check_integrity([1, 2, 3])
        with pytest.raises(TypeError):
            qc.detect_outliers([1, 2, 3], ['ghi'])

    def test_check_integrity_with_duplicates(self, sample_df, tmp_path):
        df = pd.concat([sample_df, sample_df.iloc[[0]]])
        qc = QualityControl(output_dir=str(tmp_path))
        result = qc.check_integrity(df, "Duplicate Data")
        assert result['duplicates'] == 1
        assert result['time_gaps'] == 0

    def test_detect_outliers_physical_bounds(self, sample_df, tmp_path):
        df = sample_df.copy()
        df['power'] = np.linspace(0, 100, 100)
        df.loc[df.index[50], 'power'] = 150  # 超出物理上限
        df.loc[df.index[51], 'power'] = -10  # 低于物理下限
        qc = QualityControl(output_dir=str(tmp_path))
        outliers = qc.detect_outliers(df, ['power'], physical_bounds={'power': (0, 100)})
        assert outliers.get('power', 0) == 2
