# OneHealth project - Database

(to be updated...)

Scripts for working with the database, including:

#### Input data downloading and processing

* downloading GRIB & netCDF files from [Climate Data Store](https://cds.climate.copernicus.eu/) (ERA5-Land)
* longitudes in each netCDF file are adjusted from (0-360) to (-180 to 180)
* temperatures in each netCDF file are converted from Kevin to Celsius

## Setting up CDS API
To use  CDS API for downloading data, you need to first create an account on CDS to obtain your personal access token.

Create a `.cdsapirc` file containing your personal access token by following [this instruction](https://cds.climate.copernicus.eu/how-to-api).

## Getting the NUTS regions
The regions are set [here](https://ec.europa.eu/eurostat/en/web/products-manuals-and-guidelines/w/ks-gq-23-010) and corresponding shapefiles can be downloaded [here](https://ec.europa.eu/eurostat/web/gisco/geodata/statistical-units/territorial-units-statistics).

## Run PostgreSQL database with Docker
```bash
docker compose up
```
