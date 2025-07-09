from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
import yaml
import pooch
import dotenv
import os
from onehealth_db import postgresql_database as db
import zipfile


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


def create_directories(dir: str) -> None:
    """
    Create directories if they do not exist.

    Args:
        dir (str): String of the directory to create/use.
    """
    output_dir = Path(dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def get_engine():
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


def insert_data(engine, shapefiles_folder_path):
    # check that the folder exists
    if not shapefiles_folder_path.is_dir():
        raise ValueError(
            f"Shapefile folder path {shapefiles_folder_path} does not exist."
        )
    # add NUTS definition data
    db.insert_nuts_def(engine, shapefiles_folder_path)


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


def main() -> None:
    """
    Main function to set up the production database and data lake.
    This function reads the production configuration, creates the necessary
    directories, and fetches the data from the configured sources.
    It is intended to be run as a script.
    """
    # set up production database and data lake using the provided config
    config = read_production_config()
    shapefile_folder_path = None
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
        if data["var_name"][0]["name"] == "NUTS-definition":
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
    # insert the population data


if __name__ == "__main__":
    main()
