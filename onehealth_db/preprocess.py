from typing import TypeVar, Union, Callable, Dict, Any
import xarray as xr
import numpy as np
import warnings


T = TypeVar("T", bound=Union[np.float64, xr.DataArray])


def convert_360_to_180(longitude: T) -> T:
    """Convert longitude from 0-360 to -180-180.

    Args:
        longitude (T): Longitude in 0-360 range.

    Returns:
        T: Longitude in -180-180 range.
    """
    return (longitude + 180) % 360 - 180


def adjust_longitude_360_to_180(
    dataset: xr.Dataset,
    limited_area: bool = False,
    var_name: str = "longitude",
) -> xr.Dataset:
    """Adjust longitude from 0-360 to -180-180.

    Args:
        dataset (xr.Dataset): Dataset with longitude in 0-360 range.
        limited_area (bool): Flag indicating if the dataset is a limited area.
            Default is False.
        var_name (str): Name of the longitude variable in the dataset.
            Default is "longitude".

    Returns:
        xr.Dataset: Dataset with longitude adjusted to -180-180 range.
    """
    # record attributes
    lon_attrs = dataset[var_name].attrs.copy()

    # adjust longitude
    dataset = dataset.assign_coords(
        longitude=convert_360_to_180(dataset[var_name])
    ).sortby(var_name)
    dataset[var_name].attrs = lon_attrs

    # update attributes of data variables
    for var in dataset.data_vars.keys():
        if limited_area:
            # get old attribute values
            old_lon_first_grid = dataset[var].attrs.get(
                "GRIB_longitudeOfFirstGridPointInDegrees"
            )
            old_lon_last_grid = dataset[var].attrs.get(
                "GRIB_longitudeOfLastGridPointInDegrees"
            )
            dataset[var].attrs.update(
                {
                    "GRIB_longitudeOfFirstGridPointInDegrees": convert_360_to_180(
                        old_lon_first_grid
                    ),
                    "GRIB_longitudeOfLastGridPointInDegrees": convert_360_to_180(
                        old_lon_last_grid
                    ),
                }
            )
        else:
            dataset[var].attrs.update(
                {
                    "GRIB_longitudeOfFirstGridPointInDegrees": np.float64(-179.9),
                    "GRIB_longitudeOfLastGridPointInDegrees": np.float64(180.0),
                }
            )

    return dataset


def convert_to_celsius(temperature_kelvin: T) -> T:
    """Convert temperature from Kelvin to Celsius.

    Args:
        temperature_kelvin (T): Temperature in Kelvin,
            accessed through t2m variable in the dataset.

    Returns:
        T: Temperature in Celsius.
    """
    return temperature_kelvin - 273.15


def convert_to_celsius_with_attributes(
    dataset: xr.Dataset,
    inplace: bool = False,
    var_name: str = "t2m",
) -> xr.Dataset:
    """Convert temperature from Kelvin to Celsius and keep attributes.

    Args:
        dataset (xr.Dataset): Dataset containing temperature in Kelvin.
        inplace (bool): If True, modify the original dataset.
            If False, return a new dataset. Default is False.
        var_name (str): Name of the temperature variable in the dataset.
            Default is "t2m".

    Returns:
        xr.Dataset: Dataset with temperature converted to Celsius.
    """
    if not inplace:
        dataset = dataset.copy(deep=True)

    # record attributes
    var_attrs = dataset[var_name].attrs.copy()

    # Convert temperature variable
    dataset[var_name] = convert_to_celsius(dataset[var_name])

    # Update attributes
    dataset[var_name].attrs = var_attrs
    dataset[var_name].attrs.update(
        {
            "GRIB_units": "C",
            "units": "C",
        }
    )

    return dataset


def rename_coords(dataset: xr.Dataset, coords_mapping: dict) -> xr.Dataset:
    """Rename coordinates in the dataset based on a mapping.

    Args:
        dataset (xr.Dataset): Dataset with coordinates to rename.
        coords_mapping (dict): Mapping of old coordinate names to new names.

    Returns:
        xr.Dataset: A new dataset with renamed coordinates.
    """
    coords_mapping_check = (
        isinstance(coords_mapping, dict)
        and bool(coords_mapping)
        and all(
            isinstance(old_name, str) and isinstance(new_name, str)
            for old_name, new_name in coords_mapping.items()
        )
    )
    if not coords_mapping_check:
        raise ValueError(
            "coords_mapping must be a non-empty dictionary of {old_name: new_name} pairs."
        )

    for old_name, new_name in coords_mapping.items():
        if old_name in dataset.coords:
            dataset = dataset.rename({old_name: new_name})
        else:
            warnings.warn(
                f"Coordinate '{old_name}' not found in the dataset and will be skipped.",
                UserWarning,
            )

    return dataset


def convert_m_to_mm(precipitation: T) -> T:
    """Convert precipitation from meters to millimeters.

    Args:
        precipitation (T): Precipitation in meters.

    Returns:
        T: Precipitation in millimeters.
    """
    return precipitation * 1000.0


def convert_m_to_mm_with_attributes(
    dataset: xr.Dataset, inplace: bool = False, var_name: str = "tp"
) -> xr.Dataset:
    """Convert precipitation from meters to millimeters and keep attributes.

    Args:
        dataset (xr.Dataset): Dataset containing precipitation in meters.
        inplace (bool): If True, modify the original dataset.
            If False, return a new dataset. Default is False.
        var_name (str): Name of the precipitation variable in the dataset.
            Default is "tp".

    Returns:
        xr.Dataset: Dataset with precipitation converted to millimeters.
    """
    if not inplace:
        dataset = dataset.copy(deep=True)

    # record attributes
    var_attrs = dataset[var_name].attrs.copy()

    # Convert precipitation variable
    dataset[var_name] = convert_m_to_mm(dataset[var_name])

    # Update attributes
    dataset[var_name].attrs = var_attrs
    dataset[var_name].attrs.update(
        {
            "GRIB_units": "mm",
            "units": "mm",
        }
    )

    return dataset


def downsample_resolution(
    dataset: xr.Dataset,
    new_resolution: float = 0.5,
    agg_funcs: Dict[str, str] | None = None,
    agg_map: Dict[str, Callable[[Any], float]] | None = None,
) -> xr.Dataset:
    """Downsample the resolution of a dataset.

    Args:
        dataset (xr.Dataset): Dataset to change resolution.
        new_resolution (float): New resolution in degrees. Default is 0.5.
        agg_funcs (Dict[str, str] | None): Aggregation functions for each variable.
            If None, default aggregation (i.e. mean) is used. Default is None.
        agg_map (Dict[str, Callable[[Any], float]] | None): Mapping of string
            to aggregation functions.
            If None, default mapping is used. Default is None.

    Returns:
        xr.Dataset: Dataset with changed resolution.
    """
    if new_resolution <= 0:
        raise ValueError("New resolution must be a positive number.")

    old_resolution = np.round(
        (dataset["longitude"][1] - dataset["longitude"][0]).item(), 2
    )

    if new_resolution <= old_resolution:
        raise ValueError(
            f"To downsample, degree of new resolution {new_resolution} "
            "should be greater than {old_resolution}."
        )

    weight = int(np.ceil(new_resolution / old_resolution))

    if agg_map is None:
        agg_map = {
            "mean": np.mean,
            "sum": np.sum,
            "max": np.max,
            "min": np.min,
        }
    if agg_funcs is None:
        agg_funcs = dict.fromkeys(dataset.data_vars, "mean")
    elif not isinstance(agg_funcs, dict):
        raise ValueError(
            "agg_funcs must be a dictionary of variable names and aggregation functions."
        )

    result = {}
    for var in dataset.data_vars:
        func_str = agg_funcs.get(var, "mean")
        func = agg_map.get(func_str, np.mean)

        # apply coarsening and reduction per variable
        result[var] = (
            dataset[var]
            .coarsen(longitude=weight, latitude=weight, boundary="trim")
            .reduce(func)
        )
        result[var].attrs = dataset[var].attrs.copy()

    # copy attributes of the dataset
    result_dataset = xr.Dataset(result)
    result_dataset.attrs = dataset.attrs.copy()

    return result_dataset


def align_lon_lat_with_popu_data(
    dataset: xr.Dataset,
    expected_longitude_max: np.float64 = np.float64(179.75),
) -> xr.Dataset:
    """Align longitude and latitude coordinates with population data\
    of the same resolution.
    This function is specifically designed to ensure that the
    longitude and latitude coordinates in the dataset match the expected
    values used in population data, which are:
    - Longitude: -179.75 to 179.75, 720 points
    - Latitude: 89.75 to -89.75, 360 points

    Args:
        dataset (xr.Dataset): Dataset with longitude and latitude coordinates.
        expected_longitude_max (np.float64): Expected maximum longitude
            after adjustment. Default is np.float64(179.75).

    Returns:
        xr.Dataset: Dataset with adjusted longitude and latitude coordinates.
    """
    old_longitude_min = dataset["longitude"].min().item()
    old_longitude_max = dataset["longitude"].max().item()

    # TODO: find a more general solution
    special_case = (
        np.isclose(expected_longitude_max, np.float64(179.75))
        and np.isclose(old_longitude_min, np.float64(-179.7))
        and np.isclose(old_longitude_max, np.float64(179.8))
    )
    if special_case:
        offset = expected_longitude_max - old_longitude_max

        # adjust coord values
        dataset = dataset.assign_coords(
            {
                "longitude": (dataset["longitude"] + offset).round(2),
                "latitude": (dataset["latitude"] + offset).round(2),
            }
        )

    return dataset


def upsample_resolution(
    dataset: xr.Dataset,
    new_resolution: float = 0.1,
    method_map: Dict[str, str] | None = None,
) -> xr.Dataset:
    """Upsample the resolution of a dataset.

    Args:
        dataset (xr.Dataset): Dataset to change resolution.
        new_resolution (float): New resolution in degrees. Default is 0.1.
        method_map (Dict[str, str] | None): Mapping of variable names to
            interpolation methods. If None, linear interpolation is used.
            Default is None.

    Returns:
        xr.Dataset: Dataset with changed resolution.
    """
    if new_resolution <= 0:
        raise ValueError("New resolution must be a positive number.")

    old_resolution = np.round(
        (dataset["longitude"][1] - dataset["longitude"][0]).item(), 2
    )

    if new_resolution >= old_resolution:
        raise ValueError(
            f"To upsample, degree of new resolution {new_resolution} "
            "should be smaller than {old_resolution}."
        )

    lat_min, lat_max = (
        dataset["latitude"].min().item(),
        dataset["latitude"].max().item(),
    )
    lon_min, lon_max = (
        dataset["longitude"].min().item(),
        dataset["longitude"].max().item(),
    )
    updated_lat = np.arange(lat_min, lat_max + new_resolution, new_resolution)
    updated_lon = np.arange(lon_min, lon_max + new_resolution, new_resolution)
    updated_coords = {
        "latitude": updated_lat,
        "longitude": updated_lon,
    }

    if method_map is None:
        method_map = dict.fromkeys(dataset.data_vars, "linear")
    elif not isinstance(method_map, dict):
        raise ValueError(
            "method_map must be a dictionary of variable names and interpolation methods."
        )

    # interpolate each variable
    result = {}
    for var in dataset.data_vars:
        method = method_map.get(var, "linear")
        result[var] = dataset[var].interp(**updated_coords, method=method)
        result[var].attrs = dataset[var].attrs.copy()

    # create a new dataset with the interpolated variables
    result_dataset = xr.Dataset(result)
    result_dataset.attrs = dataset.attrs.copy()

    return result_dataset


def truncate_data_from_time(
    dataset: xr.Dataset,
    start_date: Union[str, np.datetime64],
    var_name: str = "time",
) -> xr.Dataset:
    """Truncate data from a specific start date.

    Args:
        dataset (xr.Dataset): Dataset to truncate.
        start_date (Union[str, np.datetime64]): Start date for truncation.
            Format as "YYYY-MM-DD" or as a numpy datetime64 object.
        var_name (str): Name of the time variable in the dataset. Default is "time".

    Returns:
        xr.Dataset: Dataset truncated from the specified start date.
    """
    end_date = dataset[var_name].max().item()
    if isinstance(start_date, str):
        start_date = np.datetime64(start_date, "ns")
    return dataset.sel({var_name: slice(start_date, end_date)})
