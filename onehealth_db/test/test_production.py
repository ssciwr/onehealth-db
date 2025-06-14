from onehealth_db import production as prod
import pytest
from pathlib import Path
from importlib import resources
from importlib.resources.abc import Traversable


@pytest.fixture(scope="module")
def production_config() -> Traversable:
    """Fixture to provide the path to the test configuration file."""
    dict_path = (
        resources.files("onehealth_db") / "test" / "data" / "test_production_config.yml"
    )
    return dict_path


def test_read_production_config(production_config: Traversable):
    config_dict = prod.read_production_config()
    assert config_dict
    assert len(config_dict) == 2
    dict1 = config_dict["data_to_fetch"][0]
    assert dict1["var-name"] == ["t2m", "tp"]
    assert (
        dict1["filename"]
        == "era5_data_2016_2017_2t_tp_monthly_celsius_mm_resampled_0.5degree_trim.nc"
    )
    assert dict1["host"] == "heibox"
    assert dict1["description"]
    dict2 = config_dict["data_to_fetch"][1]
    assert dict2["var-name"] == ["total-population"]
    assert dict2["filename"] == "total_population_2016_2017.nc"
    assert dict2["host"] == "heibox"
    assert dict2["description"]
    config_dict = prod.read_production_config(production_config)
    assert config_dict["data_to_fetch"][0]["var-name"] == ["t2m"]
    assert (
        config_dict["data_to_fetch"][0]["filename"]
        == "era5_data_2016_01_2t_tp_monthly_celsius_mm_resampled_0.5degree_trim.nc"
    )
    assert "local" in config_dict["data_to_fetch"][0]["host"]
    config_dict = prod.read_production_config(str(production_config))
    assert config_dict


def test_get_production_data(tmp_path: Path):
    filename = "test_download.md"
    hash = "3e338f34099ae020462479ea0d6e07e2e25fdaa07e034f063be256a6539666f5"
    url = "https://heibox.uni-heidelberg.de/f/264a47e3aa484946b8f0/?dl=1"
    outputdir = tmp_path / "test_download"
    outputdir.mkdir(parents=True, exist_ok=True)
    completion_code = prod.get_production_data(url, filename, hash, outputdir)
    assert completion_code == 0
    assert (outputdir / filename).exists()


def test_create_directories(tmp_path: Path):
    prod.create_directories(str(tmp_path) + "test")
    testdir = tmp_path / "test"
    assert testdir.exists


def test_main():
    """Test the main function to ensure it runs without errors."""
    # This test will not check the actual functionality but will ensure
    # that the main function can be called without raising exceptions.
    try:
        prod.main()
    except Exception as e:
        pytest.fail(f"Main function raised an exception: {e}")
