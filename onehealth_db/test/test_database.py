import pytest
from onehealth_db import database
import pandas as pd


def check_table_name(con, table_name):
    """Check if the table name exists in the database."""
    check_name = con.execute(
        f"""
        SELECT * FROM information_schema.tables
        WHERE table_name = ?
        """,
        [table_name],  # prevent SQL injection
    ).fetchone()
    return check_name


@pytest.fixture()
def get_dataframe():
    data = {
        "column1": [1, 2, 3],
        "column2": ["A", "B", "C"],
    }
    df = pd.DataFrame(data)
    return df


def get_first_value(con, table_name):
    """Get the first value in the first column of the table."""
    first_col = con.execute(
        f"""
        SELECT column1 FROM {table_name}
        LIMIT 1
        """
    ).fetchone()
    return first_col


def test_import_data_emtpy_data():
    with pytest.raises(ValueError):
        database.import_data(data=None)


def test_import_data_none_path(get_dataframe):
    database.import_data(data=get_dataframe, db_path=None, table_name=None)
    assert database.DEFAULT_DB_PATH.exists()
    # check table name and data
    with database.duckdb.connect(str(database.DEFAULT_DB_PATH)) as con:
        check_name = check_table_name(con, database.DEFAULT_DB_PATH.stem)
        assert check_name is not None
        first_value = get_first_value(con, database.DEFAULT_DB_PATH.stem)
        assert first_value == (1,)
    # Clean up
    database.DEFAULT_DB_PATH.unlink()


def test_import_data_custom(get_dataframe, tmp_path):
    custom_db_path = tmp_path / "test" / "custom.db"
    database.import_data(
        data=get_dataframe, db_path=custom_db_path, table_name="custom_table"
    )
    assert custom_db_path.exists()
    assert custom_db_path.parent.exists()
    # check table name and data
    with database.duckdb.connect(str(custom_db_path)) as con:
        check_name = check_table_name(con, "custom_table")
        assert check_name is not None
        first_value = get_first_value(con, "custom_table")
        assert first_value == (1,)
    # Clean up
    custom_db_path.unlink()
    custom_db_path.parent.rmdir()
