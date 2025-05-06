# OneHealth project - Database

Scripts for downloading and processing GRIB & netCDF files.

TBU.

## Setting up CDS API
To use  CDS API for downloading data, you need to first create an account on CDS to obtain your personal access token.

Create a `.cdsapirc` file containing your personal access token by following [this instruction](https://cds.climate.copernicus.eu/how-to-api).

## Getting the NUTS regions
The regions are set [here](https://ec.europa.eu/eurostat/en/web/products-manuals-and-guidelines/w/ks-gq-23-010) and corresponding shapefiles can be downloaded [here](https://ec.europa.eu/eurostat/web/gisco/geodata/statistical-units/territorial-units-statistics).

## Run PostgreSQL database with Docker
```bash
docker compose up
```
