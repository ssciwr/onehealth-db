from onehealth_db import inout
import pytest
import xarray as xr
import numpy as np


def test_download_data(tmp_path):
    # empty output file path
    with pytest.raises(ValueError):
        inout.download_data(None, "test_dataset", {"param": "value"})

    # empty dataset name
    with pytest.raises(ValueError):
        inout.download_data("test_output.nc", "", {"param": "value"})

    # invalid dataset name
    with pytest.raises(ValueError):
        inout.download_data("test_output.nc", 123, {"param": "value"})

    # empty request information
    with pytest.raises(ValueError):
        inout.download_data("test_output.nc", "test_dataset", None)

    # invalid request information
    with pytest.raises(ValueError):
        inout.download_data("test_output.nc", "test_dataset", "invalid_request")

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
        "area": [2, -1, -2, 1],  # [N, W, S, E]
    }
    inout.download_data(output_file, dataset, request)
    assert output_file.exists()
    assert output_file.parent.exists()
    # Clean up
    output_file.unlink()


@pytest.fixture()
def get_data():
    data = np.random.rand(2, 3) * 1000 + 273.15
    data_array = xr.DataArray(
        data, dims=["x", "y"], coords={"x": [0, 1], "y": [0, 1, 2]}
    )
    return data_array


def test_convert_to_celsius(get_data):

    # Convert to Celsius
    celsius_array = inout.convert_to_celsius(get_data)
    expected_celsius_array = get_data - 273.15
    assert np.allclose(celsius_array.values, expected_celsius_array.values)
    assert celsius_array.dims == expected_celsius_array.dims
    assert all(
        celsius_array.coords[dim].equals(expected_celsius_array.coords[dim])
        for dim in celsius_array.dims
    )


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_save_to_netcdf(get_data, tmp_path):
    with pytest.raises(ValueError):
        inout.save_to_netcdf(get_data, None)

    file_name = tmp_path / "test_output_celsius.nc"
    inout.save_to_netcdf(get_data, file_name)
    assert file_name.exists()
    # Clean up
    file_name.unlink()


def test_get_filename():
    file_name = inout.get_filename(
        "reanalysis-era5-land-monthly-means",
        "netcdf",
        ["2025"],
        ["01", "02"],
        True,
        "era5_data",
    )
    assert file_name == "era5_data_2025_01_02_monthly_area.nc"

    file_name = inout.get_filename(
        "reanalysis-era5-land-monthly-means",
        "netcdf",
        ["2025"],
        [
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
        ],
        True,
        "era5_data",
    )
    assert file_name == "era5_data_2025_all_monthly_area.nc"

    file_name = inout.get_filename(
        "reanalysis-era5-land",
        "netcdf",
        ["2025"],
        ["01"],
        True,
        "era5_data",
    )
    assert file_name == "era5_data_2025_01_area.nc"

    file_name = inout.get_filename(
        "reanalysis-era5-land",
        "netcdf",
        ["2025"],
        ["01", "02"],
        False,
        "era5_data",
    )
    assert file_name == "era5_data_2025_01_02.nc"

    file_name = inout.get_filename(
        "reanalysis-era5-land",
        "grib",
        ["2025"],
        ["01", "02"],
        True,
        "era5_data",
    )
    assert file_name == "era5_data_2025_01_02_area.grib"
