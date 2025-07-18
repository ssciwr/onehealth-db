from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
import yaml
import pooch
import dotenv
import os
from onehealth_db import postgresql_database as db
import zipfile
import xarray as xr
from sqlalchemy import engine


def read_production_config(dict_path: str | Traversable | Path | None = None) -> dict:
    """
    Read configuration of the production database.

    Args:
        dict_path (str|Traversable): Path to the configuration dictionary.
            Defaults to None, which uses the default path.
    Returns:
        dict: Dict with configuration details for the
            production database.
    """
    if dict_path is None:
        dict_path = resources.files("onehealth_db") / "data" / "production_config.yml"
    # check if the file exists
    if isinstance(dict_path, str):
        dict_path = Path(dict_path)
    if not dict_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at {dict_path}")
    # read the configuration file
    with dict_path.open("r") as file:
        production_dict = yaml.safe_load(file)
    return production_dict


def get_production_data(url: str, filename: str, filehash: str, outputdir: Path) -> int:
    """
    Fetch data that is fed into the production database.

    Args:
    url (str): URL to fetch the data from.
        filename (str): Name of the file to be fetched.
        filehash (str): SHA256SUM hash of the file to verify integrity.
        outputdir (Path): Directory where the file will be saved.
    Returns:
        completion_code (int): Status code indicating the success or
            failure of the operation.
    """
    try:
        file = pooch.retrieve(
            url=url,
            known_hash=filehash,
            fname=filename,
            path=outputdir,
        )
    except Exception as e:
        print(f"Error fetching data: {e}")
        raise RuntimeError(f"Failed to fetch data from {url}") from e
    print(f"Data fetched and saved to {file}")
    return 0


def create_directories(dir: str) -> int:
    """
    Create directories if they do not exist.

    Args:
        dir (str): String of the directory to create/use.
    """
    output_dir = Path(dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return 0


def get_engine() -> engine.Engine:
    # get the db url from dotenv
    dotenv.load_dotenv()
    db_url = os.environ.get("DB_URL")
    try:
        # Here we drop all existing tables and create a new database
        # make sure this is not run in real production!
        engine = db.initialize_database(db_url, replace=True)
    except Exception as e:
        raise ValueError(
            "Could not initialize engine, please check \
                         your db_url {}".format(db_url)
        ) from e
    return engine


def insert_data(engine: engine.Engine, shapefiles_folder_path: Path) -> int:
    # check that the folder exists
    if not shapefiles_folder_path.is_dir():
        raise ValueError(
            f"Shapefile folder path {shapefiles_folder_path} does not exist."
        )
    # add NUTS definition data
    db.insert_nuts_def(engine, shapefiles_folder_path)
    return 0


def get_var_types_from_config(config: dict) -> list:
    """Get the variable types from the configuration file and
    place them in a dictionary."""
    var_types = []
    for data in config:
        for var_name in data["var_name"]:
            temp_dict = {
                "name": var_name["name"],
                "unit": var_name["unit"],
                "description": var_name["description"],
            }
            var_types.append(temp_dict)
    return var_types


def insert_var_values(
    engine: engine.Engine,
    era5_land_path: Path | None = None,
    isimip_path: Path | None = None,
    r0_path: Path | None = None,
) -> int:
    # somewhere we need to check that the paths are not None
    # or do not allow them to be None
    era5_ds = xr.open_dataset(era5_land_path, chunks={})
    isimip_ds = xr.open_dataset(isimip_path, chunks={})
    r0_ds = xr.open_dataset(r0_path, chunks={})
    # rechunk the dataset
    era5_ds = era5_ds.chunk({"time": 1, "latitude": 180, "longitude": 360})
    isimip_ds = isimip_ds.chunk({"time": 1, "latitude": 180, "longitude": 360})
    r0_ds = r0_ds.chunk({"time": 1, "latitude": 180, "longitude": 360})
    # add grid points
    grid_point_session = db.create_session(engine)
    db.insert_grid_points(
        grid_point_session,
        latitudes=era5_ds.latitude.to_numpy(),
        longitudes=era5_ds.longitude.to_numpy(),
    )
    grid_point_session.close()
    # add time points
    time_point_session = db.create_session(engine)
    db.insert_time_points(
        time_point_session,
        time_point_data=[
            (era5_ds.time.to_numpy(), False),
            (isimip_ds.time.to_numpy(), True),
        ],
    )  # True means yearly data
    time_point_session.close()
    # get id maps for grid, time, and variable types
    id_map_session = db.create_session(engine)
    grid_id_map, time_id_map, var_type_id_map = db.get_id_maps(id_map_session)
    id_map_session.close()
    # add t2m values
    _, _ = db.insert_var_values(
        engine, era5_ds, "t2m", grid_id_map, time_id_map, var_type_id_map
    )
    # add total precipitation values
    _, _ = db.insert_var_values(
        engine, era5_ds, "tp", grid_id_map, time_id_map, var_type_id_map
    )
    # add R0 values
    _, _ = db.insert_var_values(
        engine, r0_ds, "R0", grid_id_map, time_id_map, var_type_id_map
    )
    # add population data
    _, _ = db.insert_var_values(
        engine,
        isimip_ds,
        "total-population",
        grid_id_map,
        time_id_map,
        var_type_id_map,
        to_monthly=False,
    )
    return 0


def main() -> None:
    """
    Main function to set up the production database and data lake.
    This function reads the production configuration, creates the necessary
    directories, and fetches the data from the configured sources.
    It is intended to be run as a script.
    """
    # set up production database and data lake using the provided config
    config = read_production_config()
    era5_land_path = None
    isimip_path = None
    shapefile_folder_path = None
    r0_path = None
    # create the data lake structure if it does not exist
    for dir_name in config["datalake"].keys():
        create_directories(config["datalake"][dir_name])

    # fetch the data from the configured sources
    # this needs to be refactored for complexity
    for data in config["data_to_fetch"]:
        # check if the data is already in the data lake
        path_to_file = Path(config["datalake"]["datadir_silver"]) / data.get(
            "filename", ""
        )
        if path_to_file.is_file():
            print(
                f"File {data['filename']} already exists in the data lake, \
                    skipping download."
            )
        elif "local" in data["host"]:
            # if the host is local, we can use the local path
            data["url"] = str(Path(data["url"]).resolve())
            print(f"Using local path {data['url']} for {data['filename']}")
        elif "heibox" in data["host"]:
            # if the host is heibox, we need to use the heibox URL
            get_production_data(
                url=data["url"],
                filename=data["filename"],
                filehash=data["filehash"],
                outputdir=Path(config["datalake"]["datadir_silver"]),
            )
        if data["var_name"][0]["name"] in ["t2m", "tp"]:
            # set the path to the ERA5 data
            era5_land_path = (
                Path(config["datalake"]["datadir_silver"]) / data["filename"]
            )
            print(f"ERA5 land data path: {era5_land_path}")
        elif data["var_name"][0]["name"] == "R0":
            # set the path to the R0 data
            # this actually is gold level data
            r0_path = Path(config["datalake"]["datadir_silver"]) / data["filename"]
            print(f"R0 data path: {r0_path}")
        elif data["var_name"][0]["name"] == "total-population":
            # set the path to the ISIMIP data
            isimip_path = Path(config["datalake"]["datadir_silver"]) / data["filename"]
            print(f"ISIMIP population data path: {isimip_path}")
        elif data["var_name"][0]["name"] == "NUTS-definition":
            # extract file and set the path to the NUTS shapefiles
            shapefile_path = Path(config["datalake"]["datadir_silver"])
            shapefile_path = shapefile_path / data["filename"]
            # make sure the shapefile folder is unzipped
            shapefile_folder_path = shapefile_path.with_suffix("")
            with zipfile.ZipFile(shapefile_path, "r") as zip_ref:
                print("Extracting zip archive to {}.".format(shapefile_folder_path))
                zip_ref.extractall(shapefile_folder_path)
    # now the files are in the silver stage, feed into database
    if not shapefile_folder_path:
        raise ValueError(
            "Shapefile path could not be generated from the data to fetch."
        )
    engine = get_engine()
    # insert the NUTS shape data
    insert_data(engine=engine, shapefiles_folder_path=shapefile_folder_path)
    # insert the cartesian variables data
    var_type_session = db.create_session(engine)
    var_types = get_var_types_from_config(config=config["data_to_fetch"])
    db.insert_var_types(var_type_session, var_types)
    var_type_session.close()
    # insert the data
    insert_var_values(
        engine, era5_land_path=era5_land_path, isimip_path=isimip_path, r0_path=r0_path
    )


if __name__ == "__main__":
    main()
