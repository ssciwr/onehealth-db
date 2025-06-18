from typing import TypeVar, Union
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
    expected_longitude_max: np.float64 | None = np.float64(179.75),
) -> xr.Dataset:
    """Downsample the resolution of a dataset.

    Args:
        dataset (xr.Dataset): Dataset to change resolution.
        new_resolution (float): New resolution in degrees. Default is 0.5.
        expected_longitude_max (np.float64 | None): Expected maximum longitude
            after resolution change. If None, no further adjustment is made.
            Default is np.float64(179.75).

    Returns:
        xr.Dataset: Dataset with changed resolution.
    """
    if new_resolution <= 0:
        raise ValueError("New resolution must be a positive number.")

    old_longitude_min = dataset["longitude"].min().item()
    old_longitude_max = dataset["longitude"].max().item()

    old_resolution = np.round(
        (dataset["longitude"][1] - dataset["longitude"][0]).item(), 2
    )

    if new_resolution < old_resolution:
        raise ValueError(
            f"Degree of new resolution {new_resolution} "
            "should be greater than {old_resolution}."
        )

    weight = int(np.ceil(new_resolution / old_resolution))

    # change resolution
    dataset = dataset.coarsen(
        latitude=int(weight), longitude=int(weight), boundary="trim"
    ).mean()

    # handle floating point precision issues
    # TODO: find a more general solution
    special_case = (
        np.isclose(expected_longitude_max, np.float64(179.75))
        and np.isclose(old_longitude_min, np.float64(-179.9))
        and np.isclose(old_longitude_max, np.float64(180.0))
    )
    if special_case:
        new_longitude_max = dataset["longitude"].max().item()
        offset = expected_longitude_max - new_longitude_max

        # adjust coord values
        dataset = dataset.assign_coords(
            {
                "longitude": (dataset["longitude"] + offset).round(2),
                "latitude": (dataset["latitude"] + offset).round(2),
            }
        )
    return dataset
