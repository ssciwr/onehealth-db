import pytest
from onehealth_db import duck_database
import pandas as pd
from importlib import resources


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
        duck_database.import_data(data=None)


def test_import_data_none_path(get_dataframe):
    duck_database.import_data(data=get_dataframe, db_path=None, table_name=None)
    assert duck_database.DEFAULT_DB_PATH.exists()
    # check table name and data
    with duck_database.duckdb.connect(str(duck_database.DEFAULT_DB_PATH)) as con:
        check_name = check_table_name(con, duck_database.DEFAULT_DB_PATH.stem)
        assert check_name is not None
        first_value = get_first_value(con, duck_database.DEFAULT_DB_PATH.stem)
        assert first_value == (1,)
    # Clean up
    duck_database.DEFAULT_DB_PATH.unlink()


def test_import_data_custom(get_dataframe, tmp_path):
    custom_db_path = tmp_path / "test" / "custom.db"
    duck_database.import_data(
        data=get_dataframe, db_path=custom_db_path, table_name="custom_table"
    )
    assert custom_db_path.exists()
    assert custom_db_path.parent.exists()
    # check table name and data
    with duck_database.duckdb.connect(str(custom_db_path)) as con:
        check_name = check_table_name(con, "custom_table")
        assert check_name is not None
        first_value = get_first_value(con, "custom_table")
        assert first_value == (1,)
    # Clean up
    custom_db_path.unlink()
    custom_db_path.parent.rmdir()


def test_file_to_dataframe_invalid(tmp_path):
    # invalid file
    with pytest.raises(FileNotFoundError):
        duck_database.file_to_dataframe(tmp_path / "non_existent_file.nc")

    # unsupported file format
    invalid_file = tmp_path / "invalid_file.txt"
    invalid_file.write_text("This is not a valid file format.")

    with pytest.raises(ValueError):
        duck_database.file_to_dataframe(invalid_file)

    # empty columns
    with pytest.raises(ValueError):
        duck_database.file_to_dataframe(invalid_file, columns=[])

    # invalid col type
    with pytest.raises(ValueError):
        duck_database.file_to_dataframe(invalid_file, columns=123)

    # unsupported col
    with pytest.raises(ValueError):
        duck_database.file_to_dataframe(invalid_file, columns=["unsupported_column"])


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_file_to_dataframe_netcdf(get_dataframe, tmp_path):
    ds = get_dataframe.to_xarray()

    netcdf_file = tmp_path / "test_file.nc"
    ds.to_netcdf(netcdf_file, mode="a")
    # Convert to DataFrame
    df_netcdf = duck_database.file_to_dataframe(
        netcdf_file, columns=["column1", "column2"]
    )
    assert isinstance(df_netcdf, pd.DataFrame)
    assert df_netcdf.shape == (3, 2)
    assert "column1" in df_netcdf.columns
    assert "column2" in df_netcdf.columns
    # Clean up
    netcdf_file.unlink()
    tmp_path.rmdir()


@pytest.fixture
def get_grib_sample_path():
    pkg = resources.files("onehealth_db")
    grib_path = pkg / "test" / "data" / "era5_2025_03_monthly_area_1-1-0-m1.grib"
    return grib_path


def test_file_to_dataframe_grib_col_default(get_grib_sample_path):
    # Convert to DataFrame
    df_grib = duck_database.file_to_dataframe(get_grib_sample_path, columns="default")
    assert isinstance(df_grib, pd.DataFrame)
    assert df_grib.shape[1] == 4
    assert "valid_time" in df_grib.columns
    assert "latitude" in df_grib.columns
    assert "longitude" in df_grib.columns
    assert "t2m" in df_grib.columns
    # Clean up idx file due to cfgrib
    idx_files = get_grib_sample_path.parent.glob("*.idx")
    for idx_file in idx_files:
        idx_file.unlink()


def test_file_to_dataframe_grib_col_all(get_grib_sample_path):
    # Convert to DataFrame
    df_grib = duck_database.file_to_dataframe(get_grib_sample_path, columns="all")
    assert isinstance(df_grib, pd.DataFrame)
    # download grib file has 8 columns
    # latitude, longitude, number, time, step, surface, valid_time, t2m
    assert df_grib.shape[1] == 8
    assert "valid_time" in df_grib.columns
    assert "time" in df_grib.columns
    # Clean up idx file due to cfgrib
    idx_files = get_grib_sample_path.parent.glob("*.idx")
    for idx_file in idx_files:
        idx_file.unlink()
