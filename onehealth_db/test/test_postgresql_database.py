import pytest
from onehealth_db import postgresql_database as postdb
import numpy as np
import xarray as xr
from testcontainers.postgres import PostgresContainer
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm.session import Session, sessionmaker
import geopandas as gpd
from shapely.geometry import Polygon


# for local docker desktop,
# environ["DOCKER_HOST"] is "unix:///home/[user]/.docker/desktop/docker.sock"


@pytest.fixture(scope="module")
def get_docker_image():
    return "postgis/postgis:17-3.5"


@pytest.fixture(scope="module")
def get_engine_with_tables(get_docker_image):
    with PostgresContainer(get_docker_image) as postgres:
        engine = create_engine(postgres.get_connection_url())
        postdb.Base.metadata.create_all(engine)
        yield engine
        postdb.Base.metadata.drop_all(engine)


@pytest.fixture(scope="module")
def get_engine_without_tables(get_docker_image):
    with PostgresContainer(get_docker_image) as postgres:
        engine = create_engine(postgres.get_connection_url())
        yield engine
        postdb.Base.metadata.drop_all(engine)


@pytest.fixture(scope="function")
def get_session(get_engine_with_tables):
    connection = get_engine_with_tables.connect()
    Session = sessionmaker(bind=connection)
    session = Session()

    yield session

    # tear down
    session.close()


def cleanup(engine):
    # drop all tables
    postdb.Base.metadata.drop_all(engine)


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


def test_insert_nuts_def(get_engine_with_tables, get_session, tmp_path):
    nuts_path = tmp_path / "nuts_def.shp"
    polygon1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    polygon2 = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])
    gdf = gpd.GeoDataFrame(
        {
            "NUTS_ID": ["NUTS1", "NUTS2"],
            "LEVL_CODE": [1, 2],
            "CNTR_CODE": [1, 1],
            "NAME_LATN": ["Test NUTS", "Test NUTS2"],
            "NUTS_NAME": ["Test NUTS", "Test NUTS2"],
            "MOUNT_TYPE": [0.0, 0.0],
            "URBN_TYPE": [1.0, 1.0],
            "COAST_TYPE": [1.0, 1.0],
        },
        geometry=[polygon1, polygon2],
        crs="EPSG:4326",
    )
    gdf.to_file(nuts_path, driver="ESRI Shapefile")

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


@pytest.fixture
def get_var_type_list():
    return [
        {
            "name": "test_var",
            "unit": "1",
            "description": "Test variable",
        },
        {
            "name": "test_var2",
            "unit": "1",
            "description": "Test variable 2",
        },
    ]


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
    assert result[0].latitude == 0.0
    assert result[0].longitude == 0.0
    assert result[1].latitude == 0.0
    assert result[1].longitude == 1.0

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


@pytest.fixture
def get_time_point_lists():
    return [
        (
            np.array(
                [
                    np.datetime64("2024-01-01T00:00:00.000000000"),
                    np.datetime64("2023-01-01 00:00:00"),
                ]
            ),
            True,  # yearly
        ),
        (
            np.array(
                [
                    np.datetime64("2024-01-01T00:00:00.000000000"),
                    np.datetime64("2024-02-01 00:00:00"),
                    np.datetime64("2024-03-01 00:00:00"),
                ]
            ),
            False,  # monthly
        ),
    ]


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


@pytest.fixture()
def get_dataset():
    data = np.random.rand(2, 3, 2) * 1000 + 273.15
    data_array = xr.DataArray(
        data,
        dims=["latitude", "longitude", "time"],
        coords={
            "latitude": [10, 11],
            "longitude": [10, 11, 12],
            "time": [
                np.datetime64("2023-01-01", "ns"),
                np.datetime64("2024-01-01", "ns"),
            ],
        },
    )
    dataset = xr.Dataset({"t2m": data_array})
    return dataset


def retrieve_id_maps(session, dataset, var_type_data, to_monthly=False):
    # insert data into the database
    postdb.insert_grid_points(
        session, dataset.latitude.values, dataset.longitude.values
    )
    postdb.insert_time_points(session, [(dataset.time.values, to_monthly)])
    postdb.insert_var_types(session, var_type_data)

    # get the id maps
    grid_id_map, time_id_map, var_id_map = postdb.get_id_maps(session)

    return grid_id_map, time_id_map, var_id_map


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
        var_type.name,
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
