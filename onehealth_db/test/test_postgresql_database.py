import pytest
from onehealth_db import postgresql_database as postdb
import numpy as np
import xarray as xr
from testcontainers.postgres import PostgresContainer
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm.session import Session
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


@pytest.fixture(scope="module")
def get_session(get_engine_with_tables):
    session = postdb.create_session(get_engine_with_tables)
    yield session
    session.rollback()
    session.close()


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


def test_create_or_replace_tables(get_engine_without_tables):
    postdb.create_tables(get_engine_without_tables)
    postdb.create_or_replace_tables(get_engine_without_tables)
    missing_tables = get_missing_tables(get_engine_without_tables)
    assert not missing_tables, f"Missing tables: {missing_tables}"


@pytest.mark.skip(reason="Connection error. Check later.")
def test_initialize_database(get_docker_image):
    with PostgresContainer(get_docker_image) as postgres:
        db_url = postgres.get_connection_url()

    # initial initialization
    engine = postdb.initialize_database(db_url, replace=True)
    missing_tables = get_missing_tables(engine)
    assert not missing_tables, f"Missing tables: {missing_tables}"
    with engine.connect() as conn:
        result = conn.execute(text("SELECT postgis_full_version();"))
        version_text = result.fetchone()
        assert version_text is not None
        assert "POSTGIS=" in version_text[0]

    # initialize again without replacing
    with pytest.raises(Exception):
        postdb.initialize_database(db_url, replace=False)

    # initialize again with replacing
    engine = postdb.initialize_database(db_url, replace=True)
    missing_tables = get_missing_tables(engine)
    assert not missing_tables, f"Missing tables: {missing_tables}"

    # clean up
    postdb.Base.metadata.drop_all(engine)


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


@pytest.fixture
def get_time_points():
    return {
        np.datetime64("2024-01-01T00:00:00.000000000"): (2024, 1, 1),
        np.datetime64("2023-02-01 00:00:00"): (2023, 2, 1),
    }


def test_extract_time_point(get_time_points):
    for time_point, expected_data in get_time_points.items():
        year, month, day, _, _, _ = postdb.extract_time_point(time_point)
        assert (year, month, day) == expected_data


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


@pytest.fixture()
def get_dataset():
    data = np.random.rand(2, 3, 2) * 1000 + 273.15
    data_array = xr.DataArray(
        data,
        dims=["latitude", "longitude", "time"],
        coords={
            "latitude": [0, 1],
            "longitude": [0, 1, 2],
            "time": [
                np.datetime64("2023-01-01", "ns"),
                np.datetime64("2024-01-01", "ns"),
            ],
        },
    )
    dataset = xr.Dataset({"t2m": data_array})
    return dataset


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
