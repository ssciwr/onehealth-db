import pytest
import numpy as np
import xarray as xr
from onehealth_db import preprocess


@pytest.fixture()
def get_data():
    time_points = np.array(["2024-01-01", "2025-01-01"], dtype="datetime64")
    latitude = [0, 0.5]
    longitude = [0, 0.5, 1]
    longitude_first = np.float64(0.0)
    longitude_last = np.float64(359.9)

    # create random data for t2m and tp
    rng = np.random.default_rng(seed=42)
    data = rng.random((2, 2, 3)) * 1000 + 273.15
    data_array_t2m = xr.DataArray(
        data,
        dims=["time", "latitude", "longitude"],
        coords={"time": time_points, "latitude": latitude, "longitude": longitude},
    )

    data = rng.random((2, 2, 3)) / 1000
    data_array_precip = xr.DataArray(
        data,
        dims=["time", "latitude", "longitude"],
        coords={"time": time_points, "latitude": latitude, "longitude": longitude},
    )
    data_array_t2m.attrs = {
        "GRIB_units": "K",
        "units": "K",
        "GRIB_longitudeOfFirstGridPointInDegrees": longitude_first,
        "GRIB_longitudeOfLastGridPointInDegrees": longitude_last,
    }
    data_array_precip.attrs = {
        "GRIB_units": "m",
        "units": "m",
        "GRIB_longitudeOfFirstGridPointInDegrees": longitude_first,
        "GRIB_longitudeOfLastGridPointInDegrees": longitude_last,
    }
    return data_array_t2m, data_array_precip


@pytest.fixture()
def get_dataset(get_data):
    data_t2m = get_data[0]
    data_tp = get_data[1]
    dataset = xr.Dataset(
        {"t2m": data_t2m, "tp": data_tp},
        coords={
            "time": data_t2m.time,
            "latitude": data_t2m.latitude,
            "longitude": data_t2m.longitude,
        },
    )
    # create attributes for the dataset
    dataset.attrs.update({"GRIB_centre": "ecmf"})
    dataset["longitude"].attrs.update({"units": "degrees_east"})
    return dataset


def test_convert_360_to_180(get_data):
    # convert 360 to 180, xarray
    t2m_data = get_data[0]
    converted_array = preprocess.convert_360_to_180(t2m_data)
    expected_array = (t2m_data + 180) % 360 - 180
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


def test_adjust_longitude_360_to_180(get_dataset):
    # full area
    adjusted_dataset = preprocess.adjust_longitude_360_to_180(
        get_dataset, limited_area=False
    )
    expected_dataset = get_dataset.assign_coords(
        longitude=((get_dataset.longitude + 180) % 360 - 180)
    ).sortby("longitude")

    # check if the attributes are preserved
    assert adjusted_dataset.attrs == get_dataset.attrs
    assert adjusted_dataset["t2m"].attrs.get("units") == get_dataset["t2m"].attrs.get(
        "units"
    )
    assert adjusted_dataset["longitude"].attrs == get_dataset["longitude"].attrs
    for var in adjusted_dataset.data_vars.keys():
        assert adjusted_dataset[var].attrs.get(
            "GRIB_longitudeOfFirstGridPointInDegrees"
        ) == np.float64(-179.9)
        assert adjusted_dataset[var].attrs.get(
            "GRIB_longitudeOfLastGridPointInDegrees"
        ) == np.float64(180.0)

    # check if the data is adjusted correctly
    assert np.allclose(adjusted_dataset["t2m"].values, expected_dataset["t2m"].values)
    assert adjusted_dataset["t2m"].dims == expected_dataset["t2m"].dims
    assert all(
        adjusted_dataset["t2m"].coords[dim].equals(expected_dataset["t2m"].coords[dim])
        for dim in adjusted_dataset["t2m"].dims
    )

    # limited area
    for var in get_dataset.data_vars.keys():
        get_dataset[var].attrs.update(
            {
                "GRIB_longitudeOfFirstGridPointInDegrees": np.float64(-45.0),
                "GRIB_longitudeOfLastGridPointInDegrees": np.float64(45.0),
            }
        )
    adjusted_dataset = preprocess.adjust_longitude_360_to_180(
        get_dataset, limited_area=True
    )
    for var in adjusted_dataset.data_vars.keys():
        assert adjusted_dataset[var].attrs.get(
            "GRIB_longitudeOfFirstGridPointInDegrees"
        ) == np.float64(-45.0)
        assert adjusted_dataset[var].attrs.get(
            "GRIB_longitudeOfLastGridPointInDegrees"
        ) == np.float64(45.0)


def test_convert_to_celsius(get_data):
    t2m_data = get_data[0]
    # convert to Celsius, xarray
    celsius_array = preprocess.convert_to_celsius(t2m_data)
    expected_celsius_array = t2m_data - 273.15
    assert np.allclose(celsius_array.values, expected_celsius_array.values)
    assert celsius_array.dims == expected_celsius_array.dims
    assert all(
        celsius_array.coords[dim].equals(expected_celsius_array.coords[dim])
        for dim in celsius_array.dims
    )

    # float numbers
    kelvin_temp = 300.0
    celsius_temp = preprocess.convert_to_celsius(kelvin_temp)
    expected_temp = kelvin_temp - 273.15
    assert np.isclose(celsius_temp, expected_temp)


def test_convert_to_celsius_with_attributes_no_inplace(get_dataset):
    # convert to Celsius
    celsius_dataset = preprocess.convert_to_celsius_with_attributes(
        get_dataset, inplace=False
    )
    expected_celsius_array = get_dataset["t2m"] - 273.15

    # check if the attributes are preserved
    assert celsius_dataset.attrs == get_dataset.attrs
    assert celsius_dataset["t2m"].attrs.get("GRIB_units") == "C"
    assert celsius_dataset["t2m"].attrs.get("units") == "C"

    # check if the data is converted correctly
    assert np.allclose(celsius_dataset["t2m"].values, expected_celsius_array.values)
    assert celsius_dataset["t2m"].dims == expected_celsius_array.dims
    assert all(
        celsius_dataset["t2m"].coords[dim].equals(expected_celsius_array.coords[dim])
        for dim in celsius_dataset["t2m"].dims
    )


def test_convert_to_celsius_with_attributes_inplace(get_dataset):
    # convert to Celsius
    org_data_array = get_dataset["t2m"].copy()
    org_ds_attrs = get_dataset.attrs.copy()
    preprocess.convert_to_celsius_with_attributes(get_dataset, inplace=True)
    expected_celsius_array = org_data_array - 273.15

    # check if the attributes are preserved
    assert get_dataset.attrs == org_ds_attrs
    assert get_dataset["t2m"].attrs.get("GRIB_units") == "C"
    assert get_dataset["t2m"].attrs.get("units") == "C"

    # check if the data is converted correctly
    assert np.allclose(get_dataset["t2m"].values, expected_celsius_array.values)
    assert get_dataset["t2m"].dims == expected_celsius_array.dims
    assert all(
        get_dataset["t2m"].coords[dim].equals(expected_celsius_array.coords[dim])
        for dim in get_dataset["t2m"].dims
    )


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


def test_convert_m_to_mm(get_data):
    tp_data = get_data[1]
    # convert m to mm, xarray
    mm_array = preprocess.convert_m_to_mm(tp_data)
    expected_mm_array = tp_data * 1000.0
    assert np.allclose(mm_array.values, expected_mm_array.values)
    assert mm_array.dims == expected_mm_array.dims
    assert all(
        mm_array.coords[dim].equals(expected_mm_array.coords[dim])
        for dim in mm_array.dims
    )

    # float numbers
    m_precip = 0.001
    mm_precip = preprocess.convert_m_to_mm(m_precip)
    expected_precip = m_precip * 1000.0
    assert np.isclose(mm_precip, expected_precip)


def test_convert_m_to_mm_with_attributes_no_inplace(get_dataset):
    # convert m to mm
    mm_dataset = preprocess.convert_m_to_mm_with_attributes(
        get_dataset, inplace=False, var_name="tp"
    )
    expected_mm_array = get_dataset["tp"] * 1000.0

    # check if the attributes are preserved
    assert mm_dataset.attrs == get_dataset.attrs
    assert mm_dataset["tp"].attrs.get("GRIB_units") == "mm"
    assert mm_dataset["tp"].attrs.get("units") == "mm"

    # check if the data is converted correctly
    assert np.allclose(mm_dataset["tp"].values, expected_mm_array.values)
    assert mm_dataset["tp"].dims == expected_mm_array.dims
    assert all(
        mm_dataset["tp"].coords[dim].equals(expected_mm_array.coords[dim])
        for dim in mm_dataset["tp"].dims
    )


def test_convert_m_to_mm_with_attributes_inplace(get_dataset):
    # convert m to mm
    org_data_array = get_dataset["tp"].copy()
    org_ds_attrs = get_dataset.attrs.copy()
    preprocess.convert_m_to_mm_with_attributes(get_dataset, inplace=True, var_name="tp")
    expected_mm_array = org_data_array * 1000.0

    # check if the attributes are preserved
    assert get_dataset.attrs == org_ds_attrs
    assert get_dataset["tp"].attrs.get("GRIB_units") == "mm"
    assert get_dataset["tp"].attrs.get("units") == "mm"

    # check if the data is converted correctly
    assert np.allclose(get_dataset["tp"].values, expected_mm_array.values)
    assert get_dataset["tp"].dims == expected_mm_array.dims
    assert all(
        get_dataset["tp"].coords[dim].equals(expected_mm_array.coords[dim])
        for dim in get_dataset["tp"].dims
    )


def test_downsample_resolution_invalid(get_dataset):
    with pytest.raises(ValueError):
        preprocess.downsample_resolution(get_dataset, new_resolution=0)
    with pytest.raises(ValueError):
        preprocess.downsample_resolution(get_dataset, new_resolution=-0.5)
    with pytest.raises(ValueError):
        preprocess.downsample_resolution(get_dataset, new_resolution=0.5)
    with pytest.raises(ValueError):
        preprocess.downsample_resolution(get_dataset, new_resolution=0.2)
    with pytest.raises(ValueError):
        preprocess.downsample_resolution(
            get_dataset, new_resolution=1.0, agg_funcs="invalid"
        )


def test_downsample_resolution_default(get_dataset):
    # downsample resolution
    downsampled_dataset = preprocess.downsample_resolution(
        get_dataset, new_resolution=1.0
    )

    # check if the dimensions are reduced
    assert len(downsampled_dataset["t2m"].dims) == 3
    assert len(downsampled_dataset["tp"].dims) == 3

    # check if the coordinates are adjusted
    assert np.allclose(downsampled_dataset["t2m"].latitude.values, [0.25])
    assert np.allclose(downsampled_dataset["t2m"].longitude.values, [0.25])

    # check agg. values
    assert np.allclose(
        downsampled_dataset["t2m"].values.flatten(),
        np.mean(get_dataset["t2m"][:, :, :2], axis=(1, 2)),
    )

    # check attributes
    assert downsampled_dataset.attrs == get_dataset.attrs
    for var in downsampled_dataset.data_vars.keys():
        assert downsampled_dataset[var].attrs == get_dataset[var].attrs


def test_downsample_resolution_custom(get_dataset):
    # downsample resolution with custom aggregation functions
    agg_funcs = {
        "t2m": "mean",
        "tp": "sum",
    }
    downsampled_dataset = preprocess.downsample_resolution(
        get_dataset, new_resolution=1.0, agg_funcs=agg_funcs
    )

    # check if the dimensions are reduced
    assert len(downsampled_dataset["t2m"].dims) == 3
    assert len(downsampled_dataset["tp"].dims) == 3

    # check if the coordinates are adjusted
    assert np.allclose(downsampled_dataset["t2m"].latitude.values, [0.25])
    assert np.allclose(downsampled_dataset["t2m"].longitude.values, [0.25])

    # check agg. values
    assert np.allclose(
        downsampled_dataset["t2m"].values.flatten(),
        np.mean(get_dataset["t2m"][:, :, :2], axis=(1, 2)),
    )
    assert np.allclose(
        downsampled_dataset["tp"].values.flatten(),
        np.sum(get_dataset["tp"][:, :, :2], axis=(1, 2)),
    )

    # check attributes
    assert downsampled_dataset.attrs == get_dataset.attrs
    for var in downsampled_dataset.data_vars.keys():
        assert downsampled_dataset[var].attrs == get_dataset[var].attrs

    # custom agg map and agg funcs with missing variable
    downsampled_dataset = preprocess.downsample_resolution(
        get_dataset,
        new_resolution=1.0,
        agg_funcs={"t2m": "mean"},
        agg_map={"mean": np.mean},
    )  # tp will also use mean
    assert np.allclose(
        downsampled_dataset["tp"].values.flatten(),
        np.mean(get_dataset["tp"][:, :, :2], axis=(1, 2)),
    )


def test_align_lon_lat_with_popu_data_special_case(get_dataset):
    tmp_lat = [89.8, -89.7]
    tmp_lon = [-179.7, -179.2, 179.8]
    get_dataset = get_dataset.assign_coords(
        latitude=("latitude", tmp_lat),
        longitude=("longitude", tmp_lon),
    )
    aligned_dataset = preprocess.align_lon_lat_with_popu_data(
        get_dataset, expected_longitude_max=np.float64(179.75)
    )
    expected_lon = np.array([-179.75, -179.25, 179.75])
    expected_lat = np.array([89.75, -89.75])
    assert np.allclose(aligned_dataset["longitude"].values, expected_lon)
    assert np.allclose(aligned_dataset["latitude"].values, expected_lat)


def test_align_lon_lat_with_popu_data_other_cases(get_dataset):
    aligned_dataset = preprocess.align_lon_lat_with_popu_data(
        get_dataset, expected_longitude_max=np.float64(179.75)
    )
    assert np.allclose(
        aligned_dataset["longitude"].values, get_dataset["longitude"].values
    )
    assert np.allclose(
        aligned_dataset["latitude"].values, get_dataset["latitude"].values
    )

    tmp_lat = [89.8, -89.7]
    tmp_lon = [-179.7, -179.2, 179.8]
    get_dataset = get_dataset.assign_coords(
        latitude=("latitude", tmp_lat),
        longitude=("longitude", tmp_lon),
    )
    aligned_dataset = preprocess.align_lon_lat_with_popu_data(
        get_dataset, expected_longitude_max=np.float64(179.0)
    )
    assert np.allclose(
        aligned_dataset["longitude"].values, get_dataset["longitude"].values
    )
    assert np.allclose(
        aligned_dataset["latitude"].values, get_dataset["latitude"].values
    )


def test_upsample_resolution_invalid(get_dataset):
    with pytest.raises(ValueError):
        preprocess.upsample_resolution(get_dataset, new_resolution=0)
    with pytest.raises(ValueError):
        preprocess.upsample_resolution(get_dataset, new_resolution=-0.5)
    with pytest.raises(ValueError):
        preprocess.upsample_resolution(get_dataset, new_resolution=0.5)
    with pytest.raises(ValueError):
        preprocess.upsample_resolution(get_dataset, new_resolution=1.0)
    with pytest.raises(ValueError):
        preprocess.upsample_resolution(
            get_dataset, new_resolution=0.1, method_map="invalid"
        )


def test_upsample_resolution_default(get_dataset):
    # upsample resolution
    upsampled_dataset = preprocess.upsample_resolution(get_dataset, new_resolution=0.1)

    # check if the dimensions are increased
    assert len(upsampled_dataset["t2m"].dims) == 3
    assert len(upsampled_dataset["tp"].dims) == 3

    # check if the coordinates are adjusted
    assert np.allclose(
        upsampled_dataset["t2m"].latitude.values, np.arange(0.0, 0.6, 0.1)
    )  # example for latitude
    assert np.allclose(
        upsampled_dataset["t2m"].longitude.values, np.arange(0.0, 1.1, 0.1)
    )  # example for longitude

    # check interpolated values
    t2m_interp = upsampled_dataset["t2m"].sel(
        latitude=0.1, longitude=0.1, method="nearest"
    )
    t2m_expected = get_dataset["t2m"].interp(
        latitude=0.1, longitude=0.1, method="linear"
    )
    assert np.allclose(t2m_interp.values, t2m_expected.values)
    tp_interp = upsampled_dataset["tp"].sel(
        latitude=0.1, longitude=0.1, method="nearest"
    )
    tp_expected = get_dataset["tp"].interp(latitude=0.1, longitude=0.1, method="linear")
    assert np.allclose(tp_interp.values, tp_expected.values)

    # check attributes
    assert upsampled_dataset.attrs == get_dataset.attrs
    for var in upsampled_dataset.data_vars.keys():
        assert upsampled_dataset[var].attrs == get_dataset[var].attrs


def test_upsample_resolution_custom(get_dataset):
    # upsample resolution with custom interpolation methods
    method_map = {
        "t2m": "linear",
        "tp": "nearest",
    }
    upsampled_dataset = preprocess.upsample_resolution(
        get_dataset, new_resolution=0.1, method_map=method_map
    )

    # check interpolated values
    tp_interp = upsampled_dataset["tp"].sel(
        latitude=0.1, longitude=0.1, method="nearest"
    )
    tp_expected = get_dataset["tp"].interp(
        latitude=0.1, longitude=0.1, method="nearest"
    )
    assert np.allclose(tp_interp.values, tp_expected.values)

    # custom map with missing variable
    method_map = {
        "t2m": "linear",
    }  # tp will also use linear interpolation
    upsampled_dataset = preprocess.upsample_resolution(
        get_dataset, new_resolution=0.1, method_map=method_map
    )
    tp_interp = upsampled_dataset["tp"].sel(
        latitude=0.1, longitude=0.1, method="nearest"
    )
    tp_expected = get_dataset["tp"].interp(latitude=0.1, longitude=0.1, method="linear")
    assert np.allclose(tp_interp.values, tp_expected.values)
