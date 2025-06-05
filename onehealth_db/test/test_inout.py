from onehealth_db import inout
import pytest
import xarray as xr
import numpy as np


def test_download_data_invalid():
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


def test_download_data_valid(tmp_path):
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
        "area": [0, -1, 0, 1],  # [N, W, S, E]
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
        data,
        dims=["latitude", "longitude"],
        coords={"latitude": [0, 1], "longitude": [0, 1, 2]},
    )
    return data_array


@pytest.fixture()
def get_dataset(get_data):
    dataset = xr.Dataset({"t2m": get_data})
    # create attributes for the dataset
    attrs = {
        "GRIB_units": "K",
        "units": "K",
        "GRIB_longitudeOfFirstGridPointInDegrees": np.float64(0.0),
        "GRIB_longitudeOfLastGridPointInDegrees": np.float64(359.9),
    }
    dataset["t2m"].attrs.update(attrs)
    dataset.attrs.update({"GRIB_centre": "ecmf"})
    dataset["longitude"].attrs.update({"units": "degrees_east"})
    return dataset


def test_convert_360_to_180(get_data):
    # convert 360 to 180, xarray
    converted_array = inout.convert_360_to_180(get_data)
    expected_array = (get_data + 180) % 360 - 180
    assert np.allclose(converted_array.values, expected_array.values)
    assert converted_array.dims == expected_array.dims
    assert all(
        converted_array.coords[dim].equals(expected_array.coords[dim])
        for dim in converted_array.dims
    )

    # convert with float values
    num = 360.0
    converted_num = inout.convert_360_to_180(num)
    expected_num = (num + 180) % 360 - 180
    assert np.isclose(converted_num, expected_num)

    num = 0.0
    converted_num = inout.convert_360_to_180(num)
    expected_num = (num + 180) % 360 - 180
    assert np.isclose(converted_num, expected_num)

    num = 180.0
    converted_num = inout.convert_360_to_180(num)
    expected_num = (num + 180) % 360 - 180
    assert np.isclose(converted_num, expected_num)

    num = 90.0
    converted_num = inout.convert_360_to_180(num)
    assert np.isclose(converted_num, num)

    num = -90.0
    converted_num = inout.convert_360_to_180(num)
    assert np.isclose(converted_num, num)


def test_adjust_longitude_360_to_180_no_inplace(get_dataset):
    adjusted_dataset = inout.adjust_longitude_360_to_180(get_dataset, inplace=False)
    expected_dataset = get_dataset.assign_coords(
        longitude=((get_dataset.longitude + 180) % 360 - 180)
    ).sortby("longitude")

    # check if the attributes are preserved
    assert adjusted_dataset.attrs == get_dataset.attrs
    assert adjusted_dataset["t2m"].attrs == get_dataset["t2m"].attrs
    assert adjusted_dataset["longitude"].attrs == get_dataset["longitude"].attrs

    # check if the data is adjusted correctly
    assert np.allclose(adjusted_dataset["t2m"].values, expected_dataset["t2m"].values)
    assert adjusted_dataset["t2m"].dims == expected_dataset["t2m"].dims
    assert all(
        adjusted_dataset["t2m"].coords[dim].equals(expected_dataset["t2m"].coords[dim])
        for dim in adjusted_dataset["t2m"].dims
    )


def test_adjust_longitude_360_to_180_inplace(get_dataset):
    org_dataset = get_dataset.copy(deep=True)
    adjusted_dataset = inout.adjust_longitude_360_to_180(get_dataset, inplace=True)
    expected_dataset = org_dataset.assign_coords(
        longitude=((get_dataset.longitude + 180) % 360 - 180)
    ).sortby("longitude")

    # check if the attributes are preserved
    assert adjusted_dataset.attrs == get_dataset.attrs
    assert adjusted_dataset["t2m"].attrs == get_dataset["t2m"].attrs
    assert adjusted_dataset["longitude"].attrs == get_dataset["longitude"].attrs

    # check if the data is adjusted correctly
    assert np.allclose(adjusted_dataset["t2m"].values, expected_dataset["t2m"].values)
    assert adjusted_dataset["t2m"].dims == expected_dataset["t2m"].dims
    assert all(
        adjusted_dataset["t2m"].coords[dim].equals(expected_dataset["t2m"].coords[dim])
        for dim in adjusted_dataset["t2m"].dims
    )

    # check if the original dataset is modified
    assert np.allclose(get_dataset["t2m"].values, expected_dataset["t2m"].values)
    assert get_dataset["t2m"].dims == expected_dataset["t2m"].dims
    assert all(
        get_dataset["t2m"].coords[dim].equals(expected_dataset["t2m"].coords[dim])
        for dim in get_dataset["t2m"].dims
    )


def test_convert_to_celsius(get_data):
    # convert to Celsius
    celsius_array = inout.convert_to_celsius(get_data)
    expected_celsius_array = get_data - 273.15
    assert np.allclose(celsius_array.values, expected_celsius_array.values)
    assert celsius_array.dims == expected_celsius_array.dims
    assert all(
        celsius_array.coords[dim].equals(expected_celsius_array.coords[dim])
        for dim in celsius_array.dims
    )


def test_convert_to_celsius_with_attributes_no_inplace(get_dataset):
    # convert to Celsius, no area
    celsius_dataset = inout.convert_to_celsius_with_attributes(
        get_dataset, limited_area=False, inplace=False
    )
    expected_celsius_array = get_dataset["t2m"] - 273.15

    # check if the attributes are preserved
    assert celsius_dataset.attrs == get_dataset.attrs
    assert celsius_dataset["t2m"].attrs.get("GRIB_units") == "C"
    assert celsius_dataset["t2m"].attrs.get("units") == "C"
    assert celsius_dataset["t2m"].attrs.get(
        "GRIB_longitudeOfFirstGridPointInDegrees"
    ) == np.float64(-179.9)
    assert celsius_dataset["t2m"].attrs.get(
        "GRIB_longitudeOfLastGridPointInDegrees"
    ) == np.float64(180.0)

    # check if the data is converted correctly
    assert np.allclose(celsius_dataset["t2m"].values, expected_celsius_array.values)
    assert celsius_dataset["t2m"].dims == expected_celsius_array.dims
    assert all(
        celsius_dataset["t2m"].coords[dim].equals(expected_celsius_array.coords[dim])
        for dim in celsius_dataset["t2m"].dims
    )

    # convert to Celsius, limited area
    get_dataset["t2m"].attrs.update(
        {
            "GRIB_longitudeOfFirstGridPointInDegrees": np.float64(-45.0),
            "GRIB_longitudeOfLastGridPointInDegrees": np.float64(45.0),
        }
    )
    celsius_dataset = inout.convert_to_celsius_with_attributes(
        get_dataset, limited_area=True, inplace=False
    )
    assert celsius_dataset.attrs == get_dataset.attrs
    assert celsius_dataset["t2m"].attrs.get("GRIB_units") == "C"
    assert celsius_dataset["t2m"].attrs.get("units") == "C"
    assert celsius_dataset["t2m"].attrs.get(
        "GRIB_longitudeOfFirstGridPointInDegrees"
    ) == np.float64(-45.0)
    assert celsius_dataset["t2m"].attrs.get(
        "GRIB_longitudeOfLastGridPointInDegrees"
    ) == np.float64(45.0)


def test_convert_to_celsius_with_attributes_inplace(get_dataset):
    # convert to Celsius, no area
    org_data_array = get_dataset["t2m"].copy()
    org_ds_attrs = get_dataset.attrs.copy()
    inout.convert_to_celsius_with_attributes(
        get_dataset, limited_area=False, inplace=True
    )
    expected_celsius_array = org_data_array - 273.15

    # check if the attributes are preserved
    assert get_dataset.attrs == org_ds_attrs
    assert get_dataset["t2m"].attrs.get("GRIB_units") == "C"
    assert get_dataset["t2m"].attrs.get("units") == "C"
    assert get_dataset["t2m"].attrs.get(
        "GRIB_longitudeOfFirstGridPointInDegrees"
    ) == np.float64(-179.9)
    assert get_dataset["t2m"].attrs.get(
        "GRIB_longitudeOfLastGridPointInDegrees"
    ) == np.float64(180.0)

    # check if the data is converted correctly
    assert np.allclose(get_dataset["t2m"].values, expected_celsius_array.values)
    assert get_dataset["t2m"].dims == expected_celsius_array.dims
    assert all(
        get_dataset["t2m"].coords[dim].equals(expected_celsius_array.coords[dim])
        for dim in get_dataset["t2m"].dims
    )

    # check if the original dataset is modified
    assert np.allclose(get_dataset["t2m"].values, expected_celsius_array.values)
    assert get_dataset["t2m"].dims == expected_celsius_array.dims
    assert all(
        get_dataset["t2m"].coords[dim].equals(expected_celsius_array.coords[dim])
        for dim in get_dataset["t2m"].dims
    )

    # convert to Celsius, limited area
    get_dataset["t2m"].attrs.update(
        {
            "GRIB_longitudeOfFirstGridPointInDegrees": np.float64(-45.0),
            "GRIB_longitudeOfLastGridPointInDegrees": np.float64(45.0),
        }
    )
    inout.convert_to_celsius_with_attributes(
        get_dataset, limited_area=True, inplace=True
    )
    assert get_dataset.attrs == org_ds_attrs
    assert get_dataset["t2m"].attrs.get("GRIB_units") == "C"
    assert get_dataset["t2m"].attrs.get("units") == "C"
    assert get_dataset["t2m"].attrs.get(
        "GRIB_longitudeOfFirstGridPointInDegrees"
    ) == np.float64(-45.0)
    assert get_dataset["t2m"].attrs.get(
        "GRIB_longitudeOfLastGridPointInDegrees"
    ) == np.float64(45.0)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_save_to_netcdf(get_data, tmp_path):
    with pytest.raises(ValueError):
        inout.save_to_netcdf(get_data, None)

    file_name = tmp_path / "test_output_celsius.nc"
    inout.save_to_netcdf(get_data, file_name)
    assert file_name.exists()
    # Clean up
    file_name.unlink()


def test_get_filename_var():
    file_name = inout.get_filename(
        "reanalysis-era5-land-monthly-means",
        "netcdf",
        ["2025"],
        ["01", "02"],
        True,
        "era5_data",
        ["2m_temperature"],
    )
    assert file_name == "era5_data_2025_01_02_2t_monthly_area_raw.nc"

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
        ["2m_temperature"],
    )
    assert file_name == "era5_data_2025_all_2t_monthly_area_raw.nc"

    file_name = inout.get_filename(
        "reanalysis-era5-land",
        "netcdf",
        ["2025"],
        ["01"],
        True,
        "era5_data",
        ["2m_temperature"],
    )
    assert file_name == "era5_data_2025_01_2t_area_raw.nc"

    file_name = inout.get_filename(
        "reanalysis-era5-land",
        "netcdf",
        ["2025"],
        ["01", "02"],
        False,
        "era5_data",
        ["2m_temperature"],
    )
    assert file_name == "era5_data_2025_01_02_2t_raw.nc"

    file_name = inout.get_filename(
        "reanalysis-era5-land",
        "grib",
        ["2025"],
        ["01", "02"],
        True,
        "era5_data",
        ["2m_temperature"],
    )
    assert file_name == "era5_data_2025_01_02_2t_area_raw.grib"


def test_get_filename_vars():
    file_name = inout.get_filename(
        "reanalysis-era5-land-monthly-means",
        "netcdf",
        ["2025"],
        ["01", "02"],
        True,
        "era5_data",
        ["2m_temperature", "total_precipitation"],
    )
    assert file_name == "era5_data_2025_01_02_2t_tp_monthly_area_raw.nc"


def test_get_filename_long():
    # long vars
    var_names = [
        "2m_temperature",
        "total_precipitation",
        "surface_pressure",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "mean_sea_level_pressure",
        "total_cloud_cover",
        "low_cloud_cover",
        "medium_cloud_cover",
        "high_cloud_cover",
    ]

    file_name = inout.get_filename(
        "reanalysis-era5-land-monthly-means",
        "netcdf",
        ["2025"],
        ["01", "02"],
        True,
        "era5_data",
        var_names,
    )
    assert file_name == "era5_data_2025_01_02_2t_etc_monthly_area_raw.nc"

    # long years and long vars
    years = [str(i) for i in range(1900, 2030)]
    file_name = inout.get_filename(
        "reanalysis-era5-land-monthly-means",
        "netcdf",
        years,
        ["01", "02"],
        True,
        "era5_data",
        var_names,
    )
    assert file_name == "era5_data_1900_2029_01_02_2t_etc_monthly_area_raw.nc"

    # non-continuous years and long vars
    file_name = inout.get_filename(
        "reanalysis-era5-land-monthly-means",
        "netcdf",
        ["2020", "2023", "2021"],
        ["01", "02"],
        True,
        "era5_data",
        var_names,
    )
    assert file_name == "era5_data_2020_2021_2023_01_02_2t_etc_monthly_area_raw.nc"

    # non-continuous years with more than 5 years
    years = [str(i) for i in range(2020, 2040, 2)]
    file_name = inout.get_filename(
        "reanalysis-era5-land-monthly-means",
        "netcdf",
        years,
        ["01", "02"],
        True,
        "era5_data",
        var_names,
    )
    assert (
        file_name
        == "era5_data_2020_2022_2024_2026_2028_etc_01_02_2t_etc_monthly_area_raw.nc"
    )

    # more than 100 chars
    years = [str(i) for i in range(1900, 2030)]
    file_name = inout.get_filename(
        "reanalysis-era5-land-monthly-means",
        "netcdf",
        years,
        ["01", "02"],
        True,
        "era5_data_plus_something_very_long_to_make_the_name_longer_than_100_chars",
        var_names,
    )
    assert (
        file_name
        == "era5_data_plus_something_very_long_to_make_the_name_longer_than_100_chars_"
        "1900_2029_01_02_2t_etc_mon_etc_raw.nc"
    )
