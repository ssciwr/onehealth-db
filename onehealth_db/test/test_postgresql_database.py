import pytest
from onehealth_db import postgresql_database as postdb
import numpy as np
import xarray as xr
from testcontainers.postgres import PostgresContainer
from sqlalchemy import create_engine, text
import os


os.environ["DOCKER_HOST"] = "unix:///home/tuyen/.docker/desktop/docker.sock"


@pytest.fixture(scope="module")
def get_engine():
    with PostgresContainer("postgis/postgis:15-3.4-alpine") as postgres:
        engine = create_engine(postgres.get_connection_url())
        postdb.Base.metadata.create_all(engine)
        yield engine
        postdb.Base.metadata.drop_all(engine)


@pytest.fixture
def get_session(get_engine):
    Session = postdb.sessionmaker(bind=get_engine)
    session = Session()
    yield session
    session.rollback()
    session.close()


def test_install_postgis(get_engine):
    postdb.install_postgis(get_engine)
    # check if postgis extension is installed
    with get_engine.connect() as conn:
        result = conn.execute(text("SELECT postgis_full_version();"))
        version_text = result.fetchone()
        assert version_text is not None
        assert "POSTGIS=" in version_text[0]


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
