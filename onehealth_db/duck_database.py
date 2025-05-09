from pathlib import Path
import duckdb
import pandas as pd
import xarray as xr
from typing import Union, List, Literal


DEFAULT_DB_PATH = Path("data/onehealth.db")


def file_to_dataframe(
    file_path: Path, columns: Union[List[str], Literal["all", "default"]] = "default"
) -> pd.DataFrame:
    """Convert a netCDF/GRIB file to a pandas DataFrame.

    Args:
        file_path (Path): The path to the file.
        columns (Union[List[str], Literal["all", "default"]]):
            The columns to include in the DataFrame.
            If "all", all columns are included.
            If "default", default columns are included.
            If a list, only the specified columns are included.

    Returns:
        pd.DataFrame: The DataFrame containing the data from the file.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")

    if not columns:
        raise ValueError("Columns must be provided.")

    if columns not in ["all", "default"] and not isinstance(columns, list):
        raise ValueError("Columns must be 'all', 'default', or a list of column names.")

    if columns == "default":
        columns = ["valid_time", "latitude", "longitude", "t2m"]

    suffix = file_path.suffix.lower()

    if suffix not in [".nc", ".grib", ".grb"]:
        raise ValueError(f"Unsupported file format: {suffix}")

    engine = "netcdf4" if suffix == ".nc" else "cfgrib"
    with xr.open_dataset(file_path, engine=engine, decode_timedelta=False) as ds:
        df = ds.to_dataframe().reset_index()
        if columns == "all":
            return df
        if not all(col in df.columns for col in columns):
            raise ValueError(f"Some columns are not present in the file: {columns}")
        # Select only the specified columns
        df = df[columns]
        return df


def import_data(
    data: pd.DataFrame, db_path: Path = None, table_name: str = None
) -> None:
    """Import data from a CSV file into the DuckDB database.

    Args:
        data (pd.DataFrame): The DataFrame to import.
        db_path (Path): The path to the DuckDB database file.
        table_name (str): The name of the table to create in the database.
    """
    if data is None or not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("Data must be a non-empty pandas DataFrame.")

    if db_path is None:
        db_path = DEFAULT_DB_PATH

    # Create the database directory if it doesn't exist
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if table_name is None:
        table_name = DEFAULT_DB_PATH.stem

    # Import data to the database
    with duckdb.connect(str(db_path)) as con:
        con.sql(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM data")
        print(f"Data imported to table {table_name} in database {db_path}")
