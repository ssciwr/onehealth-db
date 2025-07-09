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
    assert dict1["var_name"][0]["name"] in ["t2m", "tp"]
    assert (
        dict1["filename"]
        == "era5_data_2016_2017_all_2t_tp_monthly_unicoords_adjlon_celsius_mm_05deg_trim.nc"
    )
    assert dict1["host"] == "heibox"
    assert dict1["description"]
    dict2 = config_dict["data_to_fetch"][1]
    assert dict2["var_name"][0]["name"] == "total-population"
    assert dict2["filename"] == "total_population_2016_2017.nc"
    assert dict2["host"] == "heibox"
    assert dict2["description"]
    # read another config file
    config_dict = prod.read_production_config(production_config)
    assert config_dict["data_to_fetch"][0]["var_name"][0]["name"] == "t2m"
    assert (
        config_dict["data_to_fetch"][0]["filename"]
        == "era5_data_2016_01_2t_tp_monthly_celsius_mm_resampled_0.5degree_trim.nc"
    )
    assert "local" in config_dict["data_to_fetch"][0]["host"]
    config_dict = prod.read_production_config(str(production_config))
    assert config_dict


def test_get_production_data(tmp_path: Path):
    filename = "test_download.md"
    filehash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    url = "https://heibox.uni-heidelberg.de/f/264a47e3aa484946b8f0/?dl=1"
    outputdir = tmp_path / "test_download"
    outputdir.mkdir(parents=True, exist_ok=True)
    completion_code = prod.get_production_data(url, filename, filehash, outputdir)
    assert completion_code == 0
    assert (outputdir / filename).exists()


def test_create_directories(tmp_path: Path):
    prod.create_directories(str(tmp_path) + "test")
    testdir = tmp_path / "test"
    assert testdir.exists


@pytest.mark.skip(reason="Skip at the moment due to requirement of running prod db.")
def test_main():
    """Test the main function to ensure it runs without errors."""
    # This test will not check the actual functionality but will ensure
    # that the main function can be called without raising exceptions.
    try:
        prod.main()
    except Exception as e:
        pytest.fail(f"Main function raised an exception: {e}")
