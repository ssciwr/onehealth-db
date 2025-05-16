import pytest
from onehealth_db import postgresql_database as postdb
import numpy as np


@pytest.fixture
def get_time_points():
    return {
        np.datetime64("2024-01-01T00:00:00.000000000"): (2024, 1, 1),
        np.datetime64("2023-02-01 00:00:00"): (2023, 2, 1),
    }


def test_extract_time_point(get_time_points):
    for time_point, expected_data in get_time_points.items():
        year, month, day = postdb.extract_time_point(time_point)
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
