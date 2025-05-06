import cdsapi
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


def save_to_netcdf(data: xr.DataArray, filename: str, encoding: dict = None):
    """Save data to a NetCDF file.

    Args:
        data (xr.DataArray): Data to be saved.
        filename (str): The name of the output NetCDF file.
        encoding (dict): Encoding options for the NetCDF file.
    """

    if not filename:
        raise ValueError("Filename must be provided.")

    data.to_netcdf(filename, encoding=encoding)  # TODO: check structure of encoding
    print("Data saved to {}".format(filename))


def get_filename(
    ds_name: str,
    data_format: str,
    years: list,
    months: list,
    has_area: bool,
    base_name: str = "era5_data",
):
    """Get file name based on dataset name, base name, years, months and area.

    Args:
        ds_name (str): Dataset name.
        data_format (str): Data format (e.g., "netcdf", "grib").
        years (list): List of years.
        months (list): List of months.
        has_area (bool): Flag indicating if area is included.
        base_name (str): Base name for the file.
            Default is "era5_data".
    Returns:
        str: Generated file name.
    """
    year_str = "_".join(years)

    if len(set(months)) != 12:
        month_str = "_".join(months)
    else:
        month_str = "all"

    if "monthly" in ds_name:
        ds_type = "_monthly"
    else:
        ds_type = ""

    file_name = base_name + "_{}_{}".format(year_str, month_str) + ds_type

    if has_area:
        file_name = file_name + "_area"

    file_ext = "grib" if data_format == "grib" else "nc"
    file_name = file_name + "." + file_ext
    return file_name


if __name__ == "__main__":
    data_format = "netcdf"  # Change to "grib" if needed
    data_folder = Path("data/in/")

    dataset = "reanalysis-era5-land-monthly-means"
    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": ["2m_temperature"],
        "year": ["2024"],
        "month": [
            "01",
            "02",
        ],
        "time": ["00:00"],
        "data_format": data_format,
        "download_format": "unarchived",
    }
    request["area"] = [90, -90, -90, 90]  # [N, W, S, E]
    file_name = get_filename(
        dataset,
        data_format,
        request["year"],
        request["month"],
        "area" in request,
        "era5_data",
    )
    output_file = data_folder / file_name

    if not output_file.exists():
        print("Downloading data...")
        download_data(output_file, dataset, request)
    else:
        print("Data already exists at {}".format(output_file))

    celsius_file_name = file_name.split(".")[0] + "_celsius.nc"
    output_celsius_file = data_folder / celsius_file_name
    with xr.open_dataset(output_file) as ds:
        print("Converting temperature to Celsius...")
        # convert temperature to Celsius
        temperature_celsius = convert_to_celsius(ds["t2m"])
        # and save to a new NetCDF file
        encoding = {
            var: {
                "zlib": True,  # Enable compression
                "complevel": 1,  # Compression level (1–9)
                "dtype": "float32",  # Use float32 to match original
            }
            for var in ds.data_vars
        }
        save_to_netcdf(temperature_celsius, output_celsius_file, encoding)
