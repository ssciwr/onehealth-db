import pytest
from onehealth_db import postgresql_database as postdb
import numpy as np
import xarray as xr
from testcontainers.postgres import PostgresContainer
from sqlalchemy import text, inspect
from sqlalchemy.orm.session import Session
import geopandas as gpd
from shapely.geometry import Polygon
import math
from fastapi import HTTPException
from conftest import cleanup, retrieve_id_maps


# for local docker desktop,
# environ["DOCKER_HOST"] is "unix:///home/[user]/.docker/desktop/docker.sock"


def test_install_postgis(get_engine_with_tables):
    postdb.install_postgis(get_engine_with_tables)
    # check if postgis extension is installed
    with get_engine_with_tables.connect() as conn:
        result = conn.execute(text("SELECT postgis_full_version();"))
        version_text = result.fetchone()
        assert version_text is not None
        assert "POSTGIS=" in version_text[0]


def test_create_session(get_engine_with_tables):
    session = postdb.create_session(get_engine_with_tables)
    assert session is not None
    assert isinstance(session, Session)
    assert session.bind is not None
    session.close()


def get_missing_tables(engine):
    inspector = inspect(engine)
    expected_tables = {"nuts_def", "grid_point", "time_point", "var_type", "var_value"}
    existing_tables = set(inspector.get_table_names(schema="public"))
    missing_tables = expected_tables - existing_tables
    return missing_tables


def test_create_tables(get_engine_without_tables):
    postdb.create_tables(get_engine_without_tables)
    missing_tables = get_missing_tables(get_engine_without_tables)
    assert not missing_tables, f"Missing tables: {missing_tables}"

    # clean up
    cleanup(get_engine_without_tables)


def test_create_or_replace_tables(get_engine_without_tables):
    postdb.create_tables(get_engine_without_tables)
    postdb.create_or_replace_tables(get_engine_without_tables)
    missing_tables = get_missing_tables(get_engine_without_tables)
    assert not missing_tables, f"Missing tables: {missing_tables}"

    # clean up
    cleanup(get_engine_without_tables)


def test_initialize_database(get_docker_image):
    with PostgresContainer(get_docker_image) as postgres:
        db_url = postgres.get_connection_url()

        # first initialization
        engine1 = postdb.initialize_database(db_url, replace=True)
        missing_tables = get_missing_tables(engine1)
        assert not missing_tables, f"Missing tables: {missing_tables}"
        with engine1.connect() as conn:
            result = conn.execute(text("SELECT postgis_full_version();"))
            version_text = result.fetchone()
            assert version_text is not None
            assert "POSTGIS=" in version_text[0]

        # add sample data to var_type table
        session = postdb.create_session(engine1)
        new_var_type = postdb.VarType(
            name="test_var",
            unit="1",
            description="Test variable",
        )
        session.add(new_var_type)
        session.commit()
        session.close()

        # initialize again without replacing
        engine2 = postdb.initialize_database(db_url, replace=False)
        # check if the data is still there
        session = postdb.create_session(engine2)
        var_types = session.query(postdb.VarType).all()
        assert len(var_types) == 1
        assert var_types[0].name == "test_var"
        session.close()

        # initialize again with replacing
        engine3 = postdb.initialize_database(db_url, replace=True)
        missing_tables = get_missing_tables(engine3)
        assert not missing_tables, f"Missing tables: {missing_tables}"
        # all tables should be empty
        session = postdb.create_session(engine3)
        assert session.query(postdb.VarType).count() == 0
        session.close()

        # clean up
        cleanup(engine1)
        cleanup(engine2)
        cleanup(engine3)


def test_insert_nuts_def(
    get_engine_with_tables, get_session, tmp_path, get_nuts_def_data
):
    nuts_path = tmp_path / "nuts_def.shp"
    gdf_nuts_data = get_nuts_def_data
    gdf_nuts_data.to_file(nuts_path, driver="ESRI Shapefile")

    postdb.insert_nuts_def(get_engine_with_tables, nuts_path)

    result = get_session.query(postdb.NutsDef).all()
    assert len(result) == 2
    assert result[0].nuts_id == "NUTS1"
    assert result[0].name_latn == "Test NUTS"
    assert result[1].nuts_id == "NUTS2"
    assert result[1].name_latn == "Test NUTS2"

    # clean up
    get_session.query(postdb.NutsDef).delete()
    get_session.commit()


def test_add_data_list(get_session):
    data_list = [
        postdb.VarType(
            name="test_var",
            unit="1",
            description="Test variable",
        ),
        postdb.VarType(
            name="test_var2",
            unit="1",
            description="Test variable 2",
        ),
    ]
    postdb.add_data_list(get_session, data_list)

    result = get_session.query(postdb.VarType).all()
    assert len(result) == 2
    assert result[0].name == "test_var"
    assert result[1].name == "test_var2"

    # clean up
    get_session.execute(text("TRUNCATE TABLE var_type RESTART IDENTITY CASCADE"))
    get_session.commit()


@pytest.mark.filterwarnings("ignore::sqlalchemy.exc.SAWarning")
def test_add_data_list_invalid(get_session, capsys):
    # non unique name
    data_list = [
        postdb.VarType(name="test_var", unit="1"),
        postdb.VarType(name="test_var", unit="1"),
    ]
    postdb.add_data_list(get_session, data_list)
    result = get_session.query(postdb.VarType).all()
    assert len(result) == 0
    captured = capsys.readouterr()
    assert "Error inserting data:" in captured.out

    # missing required fields
    data_list = [postdb.VarType(name="test_var")]
    postdb.add_data_list(get_session, data_list)
    result = get_session.query(postdb.VarType).all()
    assert len(result) == 0
    captured = capsys.readouterr()
    assert "Error inserting data:" in captured.out

    # invalid data type
    # there is no exception raised for this case
    # TODO check it again


def test_add_data_list_bulk(get_session, get_var_type_list):
    postdb.add_data_list_bulk(get_session, get_var_type_list, postdb.VarType)

    result = get_session.query(postdb.VarType).all()
    assert len(result) == 2
    assert result[0].name == "test_var"
    assert result[1].name == "test_var2"

    # clean up
    get_session.execute(text("TRUNCATE TABLE var_type RESTART IDENTITY CASCADE"))
    get_session.commit()


def test_add_data_list_bulk_empty(get_session):
    data_list = []
    postdb.add_data_list_bulk(get_session, data_list, postdb.VarType)

    result = get_session.query(postdb.VarType).all()
    assert len(result) == 0


@pytest.mark.filterwarnings("ignore::sqlalchemy.exc.SAWarning")
def test_add_data_list_bulk_invalid(get_session, capsys):
    # non unique name
    data_list = [{"name": "test_var", "unit": "1"}, {"name": "test_var", "unit": "1"}]
    postdb.add_data_list_bulk(get_session, data_list, postdb.VarType)
    captured = capsys.readouterr()
    assert "Error inserting data:" in captured.out

    # missing required fields
    data_list = [{"name": "test_var"}]
    postdb.add_data_list_bulk(get_session, data_list, postdb.VarType)
    captured = capsys.readouterr()
    assert "Error inserting data:" in captured.out

    # invalid data type
    # there is no exception raised for this case
    # TODO check it again


def test_insert_grid_points(get_session):
    latitudes = np.array([0.0, 1.0])
    longitudes = np.array([0.0, 1.0])
    postdb.insert_grid_points(get_session, latitudes, longitudes)

    result = get_session.query(postdb.GridPoint).all()
    assert len(result) == 4
    assert math.isclose(result[0].latitude, 0.0, abs_tol=1e-5)
    assert math.isclose(result[0].longitude, 0.0, abs_tol=1e-5)
    assert math.isclose(result[1].latitude, 0.0, abs_tol=1e-5)
    assert math.isclose(result[1].longitude, 1.0, abs_tol=1e-5)

    # clean up
    get_session.execute(text("TRUNCATE TABLE grid_point RESTART IDENTITY CASCADE"))
    get_session.commit()


def test_extract_time_point():
    time_points = {
        np.datetime64("2024-01-01T00:00:00.000000000"): (2024, 1, 1),
        np.datetime64("2023-02-01 00:00:00"): (2023, 2, 1),
    }
    for time_point, expected_data in time_points.items():
        year, month, day, _, _, _ = postdb.extract_time_point(time_point)
        assert (year, month, day) == expected_data


def test_extract_time_point_invalid():
    with pytest.raises(ValueError):
        postdb.extract_time_point("2024-01-01")

    with pytest.raises(ValueError):
        postdb.extract_time_point(1)


def test_get_unique_time_points(get_time_point_lists):
    unique_time_points = postdb.get_unique_time_points(get_time_point_lists)
    assert len(unique_time_points) == 24
    assert unique_time_points[0] == np.datetime64("2023-01-01", "ns")
    assert unique_time_points[-1] == np.datetime64("2024-12-01", "ns")


def test_insert_time_points(get_session, get_time_point_lists):
    postdb.insert_time_points(get_session, get_time_point_lists)

    result = get_session.query(postdb.TimePoint).all()
    assert len(result) == 24
    assert result[0].year == 2023
    assert result[0].month == 1
    assert result[0].day == 1
    assert result[-1].year == 2024
    assert result[-1].month == 12
    assert result[-1].day == 1

    # clean up
    get_session.execute(text("TRUNCATE TABLE time_point RESTART IDENTITY CASCADE"))
    get_session.commit()


def test_insert_var_types(get_session, get_var_type_list):
    postdb.insert_var_types(get_session, get_var_type_list)

    result = get_session.query(postdb.VarType).all()
    assert len(result) == 2
    assert result[0].name == "test_var"

    # clean up
    get_session.execute(text("TRUNCATE TABLE var_type RESTART IDENTITY CASCADE"))
    get_session.commit()


def test_get_id_maps(get_session, get_dataset, get_var_type_list):
    # get the id maps
    grid_id_map, time_id_map, var_id_map = retrieve_id_maps(
        get_session, get_dataset, get_var_type_list
    )

    assert len(grid_id_map) == 6
    assert grid_id_map[(10, 10)] == 1
    assert len(time_id_map) == 2
    assert time_id_map[np.datetime64("2023-01-01", "ns")] == 1
    assert len(var_id_map) == 2
    assert var_id_map["test_var"] == 1

    # clean up
    get_session.execute(text("TRUNCATE TABLE grid_point RESTART IDENTITY CASCADE"))
    get_session.execute(text("TRUNCATE TABLE time_point RESTART IDENTITY CASCADE"))
    get_session.execute(text("TRUNCATE TABLE var_type RESTART IDENTITY CASCADE"))
    get_session.commit()


def test_convert_monthly_to_yearly(get_dataset):
    assert get_dataset.sizes == {"latitude": 2, "longitude": 3, "time": 2}
    monthly_dataset = postdb.convert_yearly_to_monthly(get_dataset)
    assert monthly_dataset.sizes == {"latitude": 2, "longitude": 3, "time": 24}
    assert monthly_dataset.t2m.shape == (2, 3, 24)
    assert monthly_dataset.t2m[0, 0, 0] == get_dataset.t2m[0, 0, 0]
    assert monthly_dataset.t2m[0, 0, 1] == get_dataset.t2m[0, 0, 0]
    assert monthly_dataset.t2m[0, 0, 2] == get_dataset.t2m[0, 0, 0]
    assert monthly_dataset.t2m[0, 0, 11] == get_dataset.t2m[0, 0, 0]
    assert monthly_dataset.t2m[0, 0, 12] == get_dataset.t2m[0, 0, 1]


def test_insert_var_values_no_to_monthly(get_engine_with_tables, get_dataset):
    var_type_data = [
        {
            "name": "t2m",
            "unit": "K",
            "description": "2m temperature",
        }
    ]
    # get the id maps
    session1 = postdb.create_session(get_engine_with_tables)
    grid_id_map, time_id_map, var_id_map = retrieve_id_maps(
        session1, get_dataset, var_type_data
    )
    session1.close()

    # insert var values
    postdb.insert_var_values(
        get_engine_with_tables,
        get_dataset,
        "t2m",
        grid_id_map,
        time_id_map,
        var_id_map,
        to_monthly=False,
    )

    # check if the data is inserted correctly
    session2 = postdb.create_session(get_engine_with_tables)
    result = session2.query(postdb.VarValue).all()
    assert len(result) == 12
    assert result[0].grid_id == 1
    assert result[0].time_id == 1
    assert result[0].var_id == 1
    session2.close()

    # clean up
    postdb.Base.metadata.drop_all(get_engine_with_tables)
    postdb.Base.metadata.create_all(get_engine_with_tables)


def test_insert_var_values_no_to_monthly_no_var(
    get_engine_with_tables, get_dataset, get_var_type_list
):
    # get the id maps
    session1 = postdb.create_session(get_engine_with_tables)
    grid_id_map, time_id_map, var_id_map = retrieve_id_maps(
        session1, get_dataset, get_var_type_list
    )
    session1.close()

    # error due to no t2m var type
    with pytest.raises(ValueError):
        postdb.insert_var_values(
            get_engine_with_tables,
            get_dataset,
            "t2m",
            grid_id_map,
            time_id_map,
            var_id_map,
            to_monthly=False,
        )

    # clean up
    postdb.Base.metadata.drop_all(get_engine_with_tables)
    postdb.Base.metadata.create_all(get_engine_with_tables)


def test_insert_var_values_to_monthly(get_engine_with_tables, get_dataset):
    var_type_data = [
        {
            "name": "t2m",
            "unit": "K",
            "description": "2m temperature",
        }
    ]
    # get the id maps
    session1 = postdb.create_session(get_engine_with_tables)
    grid_id_map, time_id_map, var_id_map = retrieve_id_maps(
        session1, get_dataset, var_type_data, to_monthly=True
    )
    session1.close()

    # insert var values
    postdb.insert_var_values(
        get_engine_with_tables,
        get_dataset,
        "t2m",
        grid_id_map,
        time_id_map,
        var_id_map,
        to_monthly=True,
    )

    # check if the data is inserted correctly
    session2 = postdb.create_session(get_engine_with_tables)
    result = session2.query(postdb.VarValue).all()
    assert len(result) == 144
    assert result[0].grid_id == 1
    assert result[0].time_id == 1
    assert result[0].var_id == 1
    assert result[0].value == result[6].value  # same year, different month
    session2.close()

    # clean up
    postdb.Base.metadata.drop_all(get_engine_with_tables)
    postdb.Base.metadata.create_all(get_engine_with_tables)


def test_get_var_value(get_session):
    # sample data
    grid_point = postdb.GridPoint(latitude=10.0, longitude=20.0)
    time_point = postdb.TimePoint(year=2023, month=1, day=1)
    var_type = postdb.VarType(name="t2m", unit="K", description="2m temperature")
    var_value = postdb.VarValue(
        grid_id=1,
        time_id=1,
        var_id=1,
        value=300.0,
    )
    get_session.add(grid_point)
    get_session.add(time_point)
    get_session.add(var_type)
    get_session.add(var_value)
    get_session.commit()

    # test the function
    result = postdb.get_var_value(
        get_session,
        str(var_type.name),
        grid_point.latitude,
        grid_point.longitude,
        time_point.year,
        time_point.month,
        time_point.day,
    )
    assert result == var_value.value

    # None case
    result = postdb.get_var_value(
        get_session,
        "non_existing_var",
        grid_point.latitude,
        grid_point.longitude,
        time_point.year,
        time_point.month,
        time_point.day,
    )
    assert result is None

    # clean up
    get_session.execute(text("TRUNCATE TABLE var_value RESTART IDENTITY CASCADE"))
    get_session.execute(text("TRUNCATE TABLE var_type RESTART IDENTITY CASCADE"))
    get_session.execute(text("TRUNCATE TABLE time_point RESTART IDENTITY CASCADE"))
    get_session.execute(text("TRUNCATE TABLE grid_point RESTART IDENTITY CASCADE"))
    get_session.commit()


def test_get_time_points(get_session, get_dataset):
    # insert time points
    postdb.insert_time_points(get_session, [(get_dataset.time.values, False)])

    # test the function
    result = postdb.get_time_points(
        get_session, start_time_point=(2023, 1), end_time_point=None
    )
    assert len(result) == 1
    assert result[0].year == 2023
    assert result[0].month == 1
    assert result[0].day == 1

    result = postdb.get_time_points(
        get_session, start_time_point=(2023, 1), end_time_point=(2024, 1)
    )
    assert len(result) == 2
    assert result[0].year == 2023
    assert result[0].month == 1
    assert result[1].year == 2024
    assert result[1].month == 1

    # test with no time points
    result = postdb.get_time_points(get_session, start_time_point=(2025, 1))
    assert len(result) == 0

    # clean up
    get_session.execute(text("TRUNCATE TABLE time_point RESTART IDENTITY CASCADE"))
    get_session.commit()


def test_get_grid_points(get_session, get_dataset):
    # insert grid points
    postdb.insert_grid_points(
        get_session, get_dataset.latitude.values, get_dataset.longitude.values
    )

    # test the function
    result = postdb.get_grid_points(get_session, area=None)
    assert len(result) == 6  # 2 latitudes * 3 longitudes
    assert math.isclose(result[0].latitude, 10.0, abs_tol=1e-5)
    assert math.isclose(result[0].longitude, 10.0, abs_tol=1e-5)

    result = postdb.get_grid_points(
        get_session, area=(11.0, 10.0, 10.0, 12.0)
    )  # [N, W, S, E]
    assert len(result) == 6
    assert math.isclose(result[0].latitude, 10.0, abs_tol=1e-5)
    assert math.isclose(result[0].longitude, 10.0, abs_tol=1e-5)

    # no grid points case
    result = postdb.get_grid_points(get_session, area=(20.0, 20.0, 20.0, 20.0))
    assert len(result) == 0

    # clean up
    get_session.execute(text("TRUNCATE TABLE grid_point RESTART IDENTITY CASCADE"))
    get_session.commit()


def test_get_var_types(get_session):
    # insert var types
    var_type_data = [
        {
            "name": "t2m",
            "unit": "K",
            "description": "2m temperature",
        }
    ]
    postdb.insert_var_types(get_session, var_type_data)

    # test the function
    result = postdb.get_var_types(get_session, var_names=None)
    assert len(result) == 1
    assert result[0].name == "t2m"
    assert result[0].unit == "K"
    assert result[0].description == "2m temperature"

    result = postdb.get_var_types(get_session, var_names=["t2m"])
    assert len(result) == 1
    assert result[0].name == "t2m"

    # test with no var types
    result = postdb.get_var_types(get_session, var_names=["non_existing_var"])
    assert len(result) == 0

    # clean up
    get_session.execute(text("TRUNCATE TABLE var_type RESTART IDENTITY CASCADE"))
    get_session.commit()


def test_sort_grid_points_get_ids(get_session, get_dataset, insert_data):
    grid_points = get_session.query(postdb.GridPoint).all()
    grid_ids, latitudes, longitudes = postdb.sort_grid_points_get_ids(grid_points)
    assert len(grid_ids) == 6  # 2 latitudes * 3 longitudes
    assert latitudes == [10.0, 11.0]
    assert longitudes == [10.0, 11.0, 12.0]
    # check if the ids are correct
    assert grid_ids[1] == (0, 0)
    assert grid_ids[4] == (1, 0)


def test_get_var_values_cartesian(get_dataset, insert_data):
    # test the function
    # normal case
    ds_result = postdb.get_var_values_cartesian(
        insert_data,
        time_point=(2023, 1),
        var_name="t2m",
    )
    assert len(ds_result["latitude, longitude, var_value"]) == 6
    values = ds_result["latitude, longitude, var_value"][0]
    print(values[0])
    assert math.isclose(values[0], 10.0, abs_tol=1e-5)
    assert math.isclose(values[1], 10.0, abs_tol=1e-5)
    assert math.isclose(values[2], 1047.1060485559633, abs_tol=1e-5)
    # with default var
    ds_result = postdb.get_var_values_cartesian(
        insert_data,
        time_point=(2023, 1),
        var_name=None,
    )
    assert len(ds_result["latitude, longitude, var_value"]) == 6
    values = ds_result["latitude, longitude, var_value"][0]
    assert math.isclose(values[0], 10.0, abs_tol=1e-5)
    assert math.isclose(values[1], 10.0, abs_tol=1e-5)
    assert math.isclose(values[2], 1047.1060485559633, abs_tol=1e-5)

    # test HTTP exceptions
    # test for missing time point
    with pytest.raises(HTTPException):
        postdb.get_var_values_cartesian(
            insert_data,
            time_point=(2020, 1),
            var_name=None,
        )
    # test for missing variable name
    with pytest.raises(HTTPException):
        postdb.get_var_values_cartesian(
            insert_data,
            time_point=(2020, 1),
            var_name=["non_existing_var"],
        )


def test_get_var_values_cartesian_download(get_dataset, insert_data, tmp_path):
    # test the function
    netcdf_filename = tmp_path / "test_var_values.nc"
    postdb.get_var_values_cartesian_for_download(
        insert_data,
        start_time_point=(2023, 1),
        end_time_point=None,
        area=None,
        var_names=None,
        netcdf_file=netcdf_filename,
    )
    assert netcdf_filename.exists()
    ds_result = xr.open_dataset(netcdf_filename)
    assert len(ds_result.latitude) == 2
    assert len(ds_result.longitude) == 3
    assert len(ds_result.time) == 1
    assert ds_result.t2m.shape == (1, 2, 3)
    assert math.isclose(ds_result.t2m[0, 0, 0], get_dataset.t2m[0, 0, 0], abs_tol=1e-5)
    assert math.isclose(ds_result.t2m[0, 1, 1], get_dataset.t2m[1, 1, 0], abs_tol=1e-5)
    # remove the file after test
    netcdf_filename.unlink()
    # with end point
    postdb.get_var_values_cartesian_for_download(
        insert_data,
        start_time_point=(2023, 1),
        end_time_point=(2024, 1),
        area=None,
        var_names=None,
        netcdf_file=netcdf_filename,
    )
    assert netcdf_filename.exists()
    ds_result = xr.open_dataset(netcdf_filename)
    assert len(ds_result.latitude) == 2
    assert len(ds_result.longitude) == 3
    assert len(ds_result.time) == 2
    assert ds_result.t2m.shape == (2, 2, 3)
    # remove the file after test
    netcdf_filename.unlink()

    # with area
    postdb.get_var_values_cartesian_for_download(
        insert_data,
        start_time_point=(2023, 1),
        end_time_point=None,
        area=(11.0, 10.0, 10.0, 11.0),  # [N, W, S, E]
        var_names=None,
        netcdf_file=netcdf_filename,
    )
    assert netcdf_filename.exists()
    ds_result = xr.open_dataset(netcdf_filename)
    assert len(ds_result.latitude) == 2
    assert len(ds_result.longitude) == 2
    assert len(ds_result.time) == 1
    assert ds_result.t2m.shape == (1, 2, 2)
    # remove the file after test
    netcdf_filename.unlink()

    # with var names
    postdb.get_var_values_cartesian_for_download(
        insert_data,
        start_time_point=(2023, 1),
        end_time_point=None,
        area=None,
        var_names=["t2m"],
        netcdf_file=netcdf_filename,
    )
    assert netcdf_filename.exists()
    ds_result = xr.open_dataset(netcdf_filename)
    assert len(ds_result.latitude) == 2
    assert len(ds_result.longitude) == 3
    assert len(ds_result.time) == 1
    assert ds_result.t2m.shape == (1, 2, 3)
    # remove the file after test
    netcdf_filename.unlink()

    # none cases
    # no time points
    with pytest.raises(HTTPException):
        postdb.get_var_values_cartesian_for_download(
            insert_data,
            start_time_point=(2025, 1),
            end_time_point=None,
            area=None,
            var_names=None,
        )

    # no grid points
    with pytest.raises(HTTPException):
        postdb.get_var_values_cartesian_for_download(
            insert_data,
            start_time_point=(2023, 1),
            end_time_point=None,
            area=(20.0, 20.0, 20.0, 20.0),  # [N, W, S, E]
            var_names=None,
        )

    # no var types
    with pytest.raises(HTTPException):
        postdb.get_var_values_cartesian_for_download(
            insert_data,
            start_time_point=(2023, 1),
            end_time_point=None,
            area=None,
            var_names=["non_existing_var"],
        )


@pytest.mark.skip
def test_get_nuts_regions(
    get_engine_with_tables, get_session, tmp_path, get_nuts_def_data
):
    # create a sample NUTS shapefile
    nuts_path = tmp_path / "nuts_def.shp"
    gdf_nuts_data = get_nuts_def_data
    gdf_nuts_data.to_file(nuts_path, driver="ESRI Shapefile")

    # insert NUTS definitions
    postdb.insert_nuts_def(get_engine_with_tables, nuts_path)

    # test the function
    # normal case
    result = postdb.get_nuts_regions(get_engine_with_tables)
    assert len(result) == 2
    assert result.loc[0, "nuts_id"] == "NUTS1"  # result is a geodataframe
    assert result.loc[0, "name_latn"] == "Test NUTS"
    assert result.loc[1, "nuts_id"] == "NUTS2"
    assert result.loc[1, "name_latn"] == "Test NUTS2"

    # clean up
    get_session.query(postdb.NutsDef).delete()
    get_session.commit()


@pytest.mark.skip
def test_get_grid_ids_in_nuts(get_engine_with_tables, get_session):
    nuts_regions = gpd.GeoDataFrame(
        {
            "nuts_id": ["NUTS1", "NUTS2"],
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            ],
        },
        crs="EPSG:4326",
    )
    latitudes = np.array([0.5, 1.0, 1.5])
    longitudes = np.array([0.5, 1.0, 1.5])
    postdb.insert_grid_points(get_session, latitudes, longitudes)

    # test the function
    # normal case
    grid_ids = postdb.get_grid_ids_in_nuts(get_engine_with_tables, nuts_regions)
    assert len(grid_ids) == 6
    assert grid_ids[0] == 1
    assert grid_ids[1] == 2

    # none cases
    grid_ids = postdb.get_grid_ids_in_nuts(
        get_engine_with_tables, nuts_regions=gpd.GeoDataFrame(geometry=[])
    )
    assert len(grid_ids) == 0

    # clean up
    get_session.execute(text("TRUNCATE TABLE grid_point RESTART IDENTITY CASCADE"))
    get_session.commit()


@pytest.mark.skip
def test_get_var_values_nuts(
    get_engine_with_tables, get_session, tmp_path, get_nuts_def_data, get_dataset
):
    # create a sample NUTS shapefile
    nuts_path = tmp_path / "nuts_def.shp"
    gdf_nuts_data = get_nuts_def_data
    gdf_nuts_data.to_file(nuts_path, driver="ESRI Shapefile")

    # insert NUTS definitions
    postdb.insert_nuts_def(get_engine_with_tables, nuts_path)

    # insert grid points
    # edit grid point to match NUTS regions
    get_dataset = get_dataset.assign_coords(
        latitude=("latitude", [0.5, 1.0]),
        longitude=("longitude", [0.5, 1.0, 1.5]),
    )
    postdb.insert_grid_points(
        get_session, get_dataset.latitude.values, get_dataset.longitude.values
    )

    # insert time points
    postdb.insert_time_points(get_session, [(get_dataset.time.values, False)])

    # insert var types
    var_type_data = [
        {
            "name": "t2m",
            "unit": "K",
            "description": "2m temperature",
        }
    ]
    postdb.insert_var_types(get_session, var_type_data)

    # get the id maps
    grid_id_map, time_id_map, var_id_map = retrieve_id_maps(
        get_session, get_dataset, var_type_data
    )

    # insert var values
    postdb.insert_var_values(
        get_engine_with_tables,
        get_dataset,
        "t2m",
        grid_id_map,
        time_id_map,
        var_id_map,
        to_monthly=False,
    )

    # test the function
    # normal case
    result_dict = postdb.get_var_values_nuts(
        get_engine_with_tables,
        get_session,
        time_point=(2023, 1),
        var_name=None,
    )
    assert len(result_dict["NUTS id, var_value"]) == 2
    assert result_dict["NUTS id, var_value"][0][0] == "NUTS1"
    assert result_dict["NUTS id, var_value"][0][1] == np.mean(
        get_dataset.t2m[:, :2, 0]
    )  # dataset has coords lat lon time
    assert result_dict["NUTS id, var_value"][1][0] == "NUTS2"
    assert result_dict["NUTS id, var_value"][1][1] == np.mean(
        get_dataset.t2m[:, 1:, 0]
    )  # dataset has coords lat lon time
    # with var names
    result_dict = postdb.get_var_values_nuts(
        get_engine_with_tables,
        get_session,
        time_point=(2023, 1),
        var_name="t2m",
    )
    assert len(result_dict["NUTS id, var_value"]) == 2
    assert result_dict["NUTS id, var_value"][0][0] == "NUTS1"

    # none cases
    # no time points
    with pytest.raises(HTTPException):
        postdb.get_var_values_nuts(
            get_engine_with_tables,
            get_session,
            time_point=(2025, 1),
            var_name=None,
        )
    # no var types
    with pytest.raises(HTTPException):
        postdb.get_var_values_nuts(
            get_engine_with_tables,
            get_session,
            time_point=(2023, 1),
            var_name="non_existing_var",
        )
    # no var values
    get_session.execute(text("TRUNCATE TABLE var_value RESTART IDENTITY CASCADE"))
    get_session.commit()
    with pytest.raises(HTTPException):
        postdb.get_var_values_nuts(
            get_engine_with_tables,
            get_session,
            time_point=(2023, 1),
            var_name=None,
        )

    # clean up
    get_session.execute(text("TRUNCATE TABLE var_value RESTART IDENTITY CASCADE"))
    get_session.execute(text("TRUNCATE TABLE var_type RESTART IDENTITY CASCADE"))
    get_session.execute(text("TRUNCATE TABLE time_point RESTART IDENTITY CASCADE"))
    get_session.execute(text("TRUNCATE TABLE grid_point RESTART IDENTITY CASCADE"))
    get_session.execute(text("TRUNCATE TABLE nuts_def RESTART IDENTITY CASCADE"))
    get_session.commit()
