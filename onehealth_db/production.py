from importlib import resources
from pathlib import Path
import yaml
import pooch


def read_production_config(dict_path: str | Path | None = None) -> dict:
    """
    Read configuration of the production database.

    Args:
        dict_path (str|Path): Path to the configuration dictionary. Defaults to None,
            which uses the default path.
    Returns:
        dict: Dict with configuration details for the
            production database.
    """
    if dict_path is None:
        with resources.path("onehealth_db", "data", "production_config.yml") as path:
            dict_path = path
    if isinstance(dict_path, str):
        dict_path = Path(dict_path)
    # check if the file exists
    if not dict_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {dict_path}")
    # read the configuration file
    with dict_path.open("r") as file:
        production_dict = yaml.safe_load(file)
    return production_dict


def get_production_data(url: str, filename: str, hash: str, outputdir: Path) -> int:
    """
    Fetch data that is fed into the production database.

    Args:
    url (str): URL to fetch the data from.
        filename (str): Name of the file to be fetched.
        hash (str): SHA256SUM hash of the file to verify integrity.
        outputdir (Path): Directory where the file will be saved.
    Returns:
        completion_code (int): Status code indicating the success or
            failure of the operation.
    """
    try:
        file = pooch.retrieve(
            url=url,
            known_hash=hash,
            path=outputdir / filename,
        )
    except Exception as e:
        print(f"Error fetching data: {e}")
        raise RuntimeError(f"Failed to fetch data from {url}") from e
    print(f"Data fetched and saved to {file}")
    return 0
