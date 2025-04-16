from pathlib import Path
import duckdb
import pandas as pd


DEFAULT_DB_PATH = Path("data/onehealth.db")


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
