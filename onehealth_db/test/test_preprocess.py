import pytest
import numpy as np
import xarray as xr
from onehealth_db import preprocess


@pytest.fixture()
def get_data():
    rng = np.random.default_rng(seed=42)
    data = rng.random((2, 3)) * 1000 + 273.15
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
    converted_array = preprocess.convert_360_to_180(get_data)
    expected_array = (get_data + 180) % 360 - 180
    assert np.allclose(converted_array.values, expected_array.values)
    assert converted_array.dims == expected_array.dims
    assert all(
        converted_array.coords[dim].equals(expected_array.coords[dim])
        for dim in converted_array.dims
    )

    # convert with float values
    num = 360.0
    converted_num = preprocess.convert_360_to_180(num)
    expected_num = (num + 180) % 360 - 180
    assert np.isclose(converted_num, expected_num)

    num = 0.0
    converted_num = preprocess.convert_360_to_180(num)
    expected_num = (num + 180) % 360 - 180
    assert np.isclose(converted_num, expected_num)

    num = 180.0
    converted_num = preprocess.convert_360_to_180(num)
    expected_num = (num + 180) % 360 - 180
    assert np.isclose(converted_num, expected_num)

    num = 90.0
    converted_num = preprocess.convert_360_to_180(num)
    assert np.isclose(converted_num, num)

    num = -90.0
    converted_num = preprocess.convert_360_to_180(num)
    assert np.isclose(converted_num, num)


def test_adjust_longitude_360_to_180_no_inplace(get_dataset):
    adjusted_dataset = preprocess.adjust_longitude_360_to_180(
        get_dataset, inplace=False
    )
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
    adjusted_dataset = preprocess.adjust_longitude_360_to_180(get_dataset, inplace=True)
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
    celsius_array = preprocess.convert_to_celsius(get_data)
    expected_celsius_array = get_data - 273.15
    assert np.allclose(celsius_array.values, expected_celsius_array.values)
    assert celsius_array.dims == expected_celsius_array.dims
    assert all(
        celsius_array.coords[dim].equals(expected_celsius_array.coords[dim])
        for dim in celsius_array.dims
    )


def test_convert_to_celsius_with_attributes_no_inplace(get_dataset):
    # convert to Celsius, no area
    celsius_dataset = preprocess.convert_to_celsius_with_attributes(
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
    celsius_dataset = preprocess.convert_to_celsius_with_attributes(
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
    preprocess.convert_to_celsius_with_attributes(
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
    preprocess.convert_to_celsius_with_attributes(
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


def test_rename_coords(get_dataset):
    renamed_dataset = preprocess.rename_coords(get_dataset, {"longitude": "lon"})

    # check if the coordinates are renamed
    assert "lon" in renamed_dataset.coords
    assert "longitude" not in renamed_dataset.coords

    # check if other data is preserved
    assert np.allclose(renamed_dataset["t2m"].values, get_dataset["t2m"].values)
    assert renamed_dataset["t2m"].dims[0] == get_dataset["t2m"].dims[0]


def test_rename_coords_invalid_mapping(get_dataset):
    with pytest.raises(ValueError):
        preprocess.rename_coords(get_dataset, coords_mapping="")

    with pytest.raises(ValueError):
        preprocess.rename_coords(get_dataset, coords_mapping={})

    with pytest.raises(ValueError):
        preprocess.rename_coords(get_dataset, coords_mapping=1)

    with pytest.raises(ValueError):
        preprocess.rename_coords(get_dataset, coords_mapping={"lon": 2.2})


def test_rename_coords_notexist_coords(get_dataset):
    with pytest.warns(UserWarning):
        renamed_dataset = preprocess.rename_coords(
            get_dataset, {"notexist": "lon", "latitude": "lat"}
        )

    # check if the coordinates are not renamed
    assert "notexist" not in renamed_dataset.coords
    assert "lon" not in renamed_dataset.coords
    assert "longitude" in renamed_dataset.coords
    assert "latitude" not in renamed_dataset.coords
    assert "lat" in renamed_dataset.coords
