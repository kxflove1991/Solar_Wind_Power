import pytest
from src.config import SimulationConfig, parse_args


class TestSimulationConfig:
    def test_default_config_valid(self):
        cfg = SimulationConfig()
        assert cfg.lat == 39.0
        assert cfg.lon == 102.0
        assert cfg.year == 2020
        assert cfg.pv_capacity_mw == 100.0
        assert cfg.wind_capacity_mw == 100.0

    def test_capacity_backward_compat(self):
        cfg = SimulationConfig(capacity_mw=50.0)
        assert cfg.pv_capacity_mw == 50.0
        assert cfg.wind_capacity_mw == 50.0

    def test_invalid_lat(self):
        with pytest.raises(ValueError, match="lat"):
            SimulationConfig(lat=91.0)

    def test_invalid_lon(self):
        with pytest.raises(ValueError, match="lon"):
            SimulationConfig(lon=181.0)

    def test_invalid_year(self):
        with pytest.raises(ValueError, match="year"):
            SimulationConfig(year=1999)

    def test_negative_capacity(self):
        with pytest.raises(ValueError, match="capacity"):
            SimulationConfig(capacity_mw=-10.0)

    def test_invalid_turbine_type(self):
        with pytest.raises(ValueError, match="turbine_type"):
            SimulationConfig(turbine_type='unknown')

    def test_invalid_database(self):
        with pytest.raises(ValueError, match="database"):
            SimulationConfig(database='UnknownDB')


class TestParseArgs:
    def test_parse_default(self):
        cfg = parse_args([])
        assert cfg.lat == 39.0
        assert cfg.lon == 102.0

    def test_parse_separate_capacities(self):
        cfg = parse_args(['--pv-capacity', '50', '--wind-capacity', '100'])
        assert cfg.pv_capacity_mw == 50.0
        assert cfg.wind_capacity_mw == 100.0

    def test_parse_legacy_capacity(self):
        cfg = parse_args(['--capacity', '80'])
        assert cfg.pv_capacity_mw == 80.0
        assert cfg.wind_capacity_mw == 80.0

    def test_parse_invalid_lat(self):
        with pytest.raises(SystemExit):
            parse_args(['--lat', '999'])
