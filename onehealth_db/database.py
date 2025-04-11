import cdsapi
import os
from pathlib import Path
import xarray as xr


def download_data(output_file: Path, dataset: str, request: dict):
    """Download data from Copernicus Climate Data Store (CDS) using the cdsapi.

    Args:
        output_file (Path): The path to the output file where data will be saved.
        dataset (str): The name of the dataset to download.
        request (dict): A dictionary containing the request parameters.
    """
    if not output_file:
        raise ValueError("Output file path must be provided.")

    if not dataset or not isinstance(dataset, str):
        raise ValueError("Dataset name must be a non-empty string.")

    if not request or not isinstance(request, dict):
        raise ValueError("Request information must be a dictionary.")

    if not output_file.exists():
        # create the directory if it doesn't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)

    client = cdsapi.Client()
    client.retrieve(dataset, request, target=str(output_file))
    print("Data downloaded successfully to {}".format(output_file))


def convert_to_celsius(temperature_kelvin: xr.DataArray) -> xr.DataArray:
    """Convert temperature from Kelvin to Celsius.

    Args:
        temperature_kelvin (xr.DataArray): Temperature in Kelvin,
            accessed through t2m variable in the dataset.

    Returns:
        xr.DataArray: Temperature in Celsius.
    """
    return temperature_kelvin - 273.15


def save_to_netcdf(data: xr.DataArray, filename: str, encoding: dict):
    """Save data to a NetCDF file.

    Args:
        data (xr.DataArray): Data to be saved.
        filename (str): The name of the output NetCDF file.
        encoding (dict): Encoding options for the NetCDF file.
    """

    if not filename:
        raise ValueError("Filename must be provided.")

    data.to_netcdf(filename, encoding=encoding)
    print("Data saved to {}".format(filename))


if __name__ == "__main__":
    data_format = "netcdf"  # Change to "grib" if needed
    file_ext = "grib" if data_format == "grib" else "nc"
    data_folder = Path("data/in/")
    file_name = "era5_data_2025_03_monthly.{}".format(file_ext)
    output_file = data_folder / file_name

    dataset = "reanalysis-era5-land-monthly-means"
    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": ["2m_temperature"],
        "year": ["2025"],
        "month": ["03"],
        "time": ["00:00"],
        "data_format": data_format,
        "download_format": "unarchived",
    }

    if not output_file.exists():
        print("Downloading data...")
        download_data(output_file, dataset, request)
    else:
        print("Data already exists at {}".format(output_file))

    # load data into xarray
    ds = xr.open_dataset(output_file)
    print("Variables in the dataset:")
    print(ds.variables)
    print("Encoding: {}".format(ds["t2m"].encoding))
    # convert temperature to Celsius
    temperature_celsius = convert_to_celsius(ds["t2m"])
    # and save to a new NetCDF file
    celsius_file_name = file_name.split(".")[0] + "_celsius.nc"
    output_celsius_file = data_folder / celsius_file_name
    encoding = {
        var: {
            "zlib": True,  # Enable compression
            "complevel": 1,  # Compression level (1–9) TODO: check info
            "dtype": "float32",  # Use float32 to match original
        }
        for var in ds.data_vars
    }
    save_to_netcdf(temperature_celsius, output_celsius_file, encoding)
