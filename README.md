# OneHealth project - Database

(to be updated...)

Scripts for working with the database, including:

#### Input data downloading and processing

* downloading GRIB & netCDF files from [Climate Data Store](https://cds.climate.copernicus.eu/) (ERA5-Land)
* longitudes in each netCDF file are adjusted from (0-360) to (-180 to 180)
* temperatures in each netCDF file are converted from Kelvin to Celsius

## Setting up CDS API for downloading ERA5-Land data
To use  CDS API for downloading data, you need to first create an account on CDS to obtain your personal access token.

Create a `.cdsapirc` file containing your personal access token by following [this instruction](https://cds.climate.copernicus.eu/how-to-api).

### Naming convention for downloaded netCDF files
Will be refactored later. See [issue #8](https://github.com/ssciwr/onehealth-db/issues/8)

File name's structure:
```
source_name_list_of_years_list_of_months_list_of_vars[_montly][_area]_raw.ext
```
* `source_name` is `"era5_data"`,
* All years are firstly sorted.
    * If years are continuous values, `list_of_years`is a concatenate of `min` and `max` values. Otherwises, `list_of_years` is a join of all years.
    * However, if there are more than 5 years, we only keep the first 5 years and replace the rest by `"_etc"`
* `list_of_months`can be `"all"`, representing a whole year, or a join of all months
* Each variable has an abbreviation derived by the first letter of each word in the variable name (e.g. `tp` for `total precipitation`).
    * All abbreviations are then concatenated
    * If this concatenated string is longer than 30 characters, we only keep the first 2 characters and replace the the rest by `"_etc"`
* If the file was downloaded from a monthly dataset, `"_monthly"` is presented in the file name
* If the downloaded data is only for an area of the grid (instead of the whole map), `"_area"` is inserted into the file name
* If the part before `"_raw"` is longer than 100 characters, only the first 100 characters are kept and the rest is replaced by `"_etc"`
* `"_raw"` is added at the end to indicate that the file is raw data
* Extension `ext` of the file can be `.nc` or `.grib`


## Population data from ISIMIP
Perform the following steps to download population data used for this project:
* go to [ISIMIP website](https://data.isimip.org/)
* search `population` from the search bar
* choose simulation round `ISIMIP3a`
* click `Input Data` -> `Direct human forcing` -> `Population data` -> `histsoc`
* choose `population_histsoc_30arcmin_annual`
* download file `population_histsoc_30arcmin_annual_1901_2021.nc`


## Getting the NUTS regions
The regions are set [here](https://ec.europa.eu/eurostat/en/web/products-manuals-and-guidelines/w/ks-gq-23-010) and corresponding shapefiles can be downloaded [here](https://ec.europa.eu/eurostat/web/gisco/geodata/statistical-units/territorial-units-statistics).

In this project, we use `EPSG: 4326`

**Note**:
* This NUTS definition file is only for Europe.
* If a country does not have NUTS level $x \in [1,3]$, the corresponding data for these levels is excluded from the shapefile.

#### `NUTS_ID` explaination:
* Structure of `NUTS_ID`: `<country><level>`
* `country`: 2 letters, representing name of a country, e.g. DE
* `level`: 0 to 3 letters or numbers, signifying the level of the NUTS region


## Run PostgreSQL database with Docker
```bash
docker compose up -d
```
Use option `-d` to run the docker service in background.

## Data flowchart
Downloaded data files will go through preprocessing steps as follows:
![data-flow.jpg](/docs/source/_static/onehealth_data_flow.jpg)