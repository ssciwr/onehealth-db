from typing import TypeVar, Union
import xarray as xr
import numpy as np


T = TypeVar("T", bound=Union[float, xr.DataArray])


def convert_360_to_180(longitude: T) -> T:
    """Convert longitude from 0-360 to -180-180.

    Args:
        longitude (T): Longitude in 0-360 range.

    Returns:
        T: Longitude in -180-180 range.
    """
    return (longitude + 180) % 360 - 180


def adjust_longitude_360_to_180(
    dataset: xr.Dataset, inplace: bool = True
) -> xr.Dataset:
    """Adjust longitude from 0-360 to -180-180.

    Args:
        dataset (xr.Dataset): Dataset with longitude in 0-360 range.
        inplace (bool): If True, modify the original dataset.
            If False, return a new dataset. Default is True.

    Returns:
        xr.Dataset: Dataset with longitude adjusted to -180-180 range.
    """
    if not inplace:
        dataset = dataset.copy(deep=True)

    # record attributes
    lon_attrs = dataset["longitude"].attrs.copy()

    # adjust longitude
    dataset = dataset.assign_coords(
        longitude=convert_360_to_180(dataset["longitude"])
    ).sortby("longitude")
    dataset["longitude"].attrs = lon_attrs

    return dataset


def convert_to_celsius(temperature_kelvin: xr.DataArray) -> xr.DataArray:
    """Convert temperature from Kelvin to Celsius.

    Args:
        temperature_kelvin (xr.DataArray): Temperature in Kelvin,
            accessed through t2m variable in the dataset.

    Returns:
        xr.DataArray: Temperature in Celsius.
    """
    return temperature_kelvin - 273.15


def convert_to_celsius_with_attributes(
    dataset: xr.Dataset, limited_area: bool = True, inplace: bool = True
) -> xr.Dataset:
    """Convert temperature from Kelvin to Celsius and keep attributes.

    Args:
        dataset (xr.Dataset): Dataset containing temperature in Kelvin.
        limited_area (bool): Flag indicating if the dataset is a limited area.
            Default is True.
        inplace (bool): If True, modify the original dataset.
            If False, return a new dataset. Default is True.

    Returns:
        xr.Dataset: Dataset with temperature converted to Celsius.
    """
    if not inplace:
        dataset = dataset.copy(deep=True)

    # record attributes
    t2m_attrs = dataset["t2m"].attrs.copy()

    # Convert temperature variable
    dataset["t2m"] = convert_to_celsius(dataset["t2m"])

    # Update attributes
    dataset["t2m"].attrs = t2m_attrs
    dataset["t2m"].attrs.update(
        {
            "GRIB_units": "C",
            "units": "C",
        }
    )

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
