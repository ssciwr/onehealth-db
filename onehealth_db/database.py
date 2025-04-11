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


if __name__ == "__main__":
    data_format = "grib"  # Change to "grib" if needed
    file_ext = "grib" if data_format == "grib" else "nc"
    output_file = Path("data/in/") / "era5_data_2025_03_monthly.{}".format(file_ext)

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
