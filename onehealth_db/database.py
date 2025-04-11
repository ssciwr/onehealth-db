import cdsapi
import os
from pathlib import Path
import xarray as xr
from user_key import USER_KEY


os.environ["CDSAPI_URL"] = "https://cds.climate.copernicus.eu/api"
os.environ["CDSAPI_KEY"] = USER_KEY


def download_data(output_file, data_format):
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

    client = cdsapi.Client()
    client.retrieve(dataset, request, target=str(output_file))
    print("Data downloaded successfully to {}".format(output_file))


if __name__ == "__main__":
    data_format = "netcdf"  # Change to "grib" if needed
    file_ext = "grib" if data_format == "grib" else "nc"
    output_file = Path("data/in/") / "era5_data_2025_03_monthly.{}".format(file_ext)

    if not output_file.exists():
        # create the directory if it doesn't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading data...")
        download_data(output_file, data_format)
    else:
        print("Data already exists at {}".format(output_file))
