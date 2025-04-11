from onehealth_db import database
import pytest


def test_download_data(tmp_path):
    # empty output file path
    with pytest.raises(ValueError):
        database.download_data(None, "test_dataset", {"param": "value"})

    # empty dataset name
    with pytest.raises(ValueError):
        database.download_data("test_output.nc", "", {"param": "value"})

    # invalid dataset name
    with pytest.raises(ValueError):
        database.download_data("test_output.nc", 123, {"param": "value"})

    # empty request information
    with pytest.raises(ValueError):
        database.download_data("test_output.nc", "test_dataset", None)

    # invalid request information
    with pytest.raises(ValueError):
        database.download_data("test_output.nc", "test_dataset", "invalid_request")

    # valid case
    output_file = tmp_path / "test" / "test_output.nc"
    dataset = "reanalysis-era5-land-monthly-means"
    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": ["2m_temperature"],
        "year": ["2025"],
        "month": ["03"],
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    database.download_data(output_file, dataset, request)
    assert output_file.exists()
    assert output_file.parent.exists()
