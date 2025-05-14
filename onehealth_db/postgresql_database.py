from sqlalchemy import (
    create_engine,
    text,
    Float,
    String,
    Integer,
    BigInteger,
    Index,
    ForeignKey,
    UniqueConstraint,
    ForeignKeyConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from geoalchemy2 import Geometry, WKBElement
from sqlalchemy.orm import sessionmaker
import geopandas as gpd
from pathlib import Path
import pandas as pd
import numpy as np


# Base declarative class
class Base(DeclarativeBase):
    pass


class NutsDef(Base):
    __tablename__ = "nuts_def"

    nuts_id: Mapped[String] = mapped_column(String(), primary_key=True)
    levl_code: Mapped[int] = mapped_column(Integer(), nullable=True)
    cntr_code: Mapped[String] = mapped_column(String(), nullable=True)
    name_latn: Mapped[String] = mapped_column(String(), nullable=True)
    nuts_name: Mapped[String] = mapped_column(String(), nullable=True)
    mount_type: Mapped[Float] = mapped_column(Float(), nullable=True)
    urbn_type: Mapped[Float] = mapped_column(Float(), nullable=True)
    coast_type: Mapped[Float] = mapped_column(Float(), nullable=True)
    geometry: Mapped[WKBElement] = mapped_column(
        Geometry(geometry_type="POINT", srid=4326)
    )


class GridPoint(Base):
    __tablename__ = "grid_point"

    id: Mapped[int] = mapped_column(Integer(), primary_key=True, autoincrement=True)
    latitude: Mapped[Float] = mapped_column(Float())
    longitude: Mapped[Float] = mapped_column(Float())

    # Geometry column for PostGIS
    point: Mapped[Geometry] = mapped_column(Geometry("POINT", srid=4326))

    __table_args__ = (
        Index("idx_point_gridpoint", "point", postgresql_using="gist"),
        UniqueConstraint("latitude", "longitude", name="uq_lat_lon"),
    )

    def __init__(self, latitude, longitude, **kw):
        super().__init__(**kw)
        self.latitude = latitude
        self.longitude = longitude
        self.point = f"SRID=4326;POINT({self.longitude} {self.latitude})"


class TimePoint(Base):
    __tablename__ = "time_point"

    id: Mapped[int] = mapped_column(Integer(), primary_key=True, autoincrement=True)
    year: Mapped[int] = mapped_column(Integer())
    month: Mapped[int] = mapped_column(Integer())
    day: Mapped[int] = mapped_column(Integer())

    __table_args__ = (
        UniqueConstraint("year", "month", "day", name="uq_year_month_day"),
    )


class VarType(Base):
    __tablename__ = "var_type"

    id: Mapped[int] = mapped_column(Integer(), primary_key=True, autoincrement=True)
    name: Mapped[String] = mapped_column(String())
    unit: Mapped[String] = mapped_column(String())
    description: Mapped[String] = mapped_column(String(), nullable=True)

    __table_args__ = (UniqueConstraint("name", name="uq_var_name"),)


class VarValue(Base):
    __tablename__ = "var_value"

    id: Mapped[int] = mapped_column(BigInteger(), primary_key=True, autoincrement=True)
    grid_id: Mapped[int] = mapped_column(Integer(), ForeignKey("grid_point.id"))
    time_id: Mapped[int] = mapped_column(Integer(), ForeignKey("time_point.id"))
    var_id: Mapped[int] = mapped_column(Integer(), ForeignKey("var_type.id"))
    value: Mapped[Float] = mapped_column(Float())

    __table_args__ = (
        UniqueConstraint("time_id", "grid_id", "var_id", name="uq_time_grid_var"),
        ForeignKeyConstraint(
            ["grid_id"],
            ["grid_point.id"],
            name="fk_grid_id",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["time_id"],
            ["time_point.id"],
            name="fk_time_id",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["var_id"],
            ["var_type.id"],
            name="fk_var_id",
            ondelete="CASCADE",
        ),
    )


def install_postgis(engine):
    """
    Install PostGIS extension on the database.
    """
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
        print("PostGIS extension installed.")


def create_session(engine):
    """
    Create a new session for the database.
    """
    Session = sessionmaker(bind=engine)
    print("Session created.")
    return Session()


def create_tables(engine):
    """
    Create tables in the database.
    """
    Base.metadata.create_all(engine)
    print("All tables created.")


def create_or_replace_tables(engine):
    """
    Create or replace tables in the database.
    """
    Base.metadata.drop_all(engine)
    print("All tables dropped.")
    create_tables(engine)


def initialize_database(db_url: str, replace: bool = False):
    """
    Initialize the database by creating the engine and tables, and installing PostGIS.
    If replace is True, it will drop and recreate the tables.
    """
    # create engine
    engine = create_engine(db_url)  # remove echo=True to show just errors in terminal

    # install PostGIS extension
    install_postgis(engine)

    # create or replace tables
    if replace:
        create_or_replace_tables(engine)
    else:
        create_tables(engine)

    print("Database initialized successfully.")

    return engine


def insert_nuts_def(engine, shapefile_path: Path):
    """
    Insert NUTS definition data into the database.
    """
    nuts_data = gpd.GeoDataFrame.from_file(shapefile_path)
    # rename columns to match the database schema
    nuts_data = nuts_data.rename(
        columns={
            "NUTS_ID": "nuts_id",
            "LEVL_CODE": "levl_code",
            "CNTR_CODE": "cntr_code",
            "NAME_LATN": "name_latn",
            "NUTS_NAME": "nuts_name",
            "MOUNT_TYPE": "mount_type",
            "URBN_TYPE": "urbn_type",
            "COAST_TYPE": "coast_type",
        }
    )
    nuts_data.to_postgis(
        NutsDef.__tablename__, engine, if_exists="replace", index=False
    )


def add_data_list(engine, data_list: list):
    """
    Add a list of data to the database.
    """
    session = create_session(engine)
    session.add_all(data_list)
    session.commit()
    session.close()


def insert_grid_points(engine, latitudes: list, longitudes: list):
    """
    Insert grid points into the database.
    """
    grid_points = [
        GridPoint(latitude=lat, longitude=lon)
        for lat, lon in zip(latitudes, longitudes)
    ]
    add_data_list(engine, grid_points)
    print("Grid points inserted.")


def extract_time_point(time_point: np.datetime64) -> tuple[int, int, int]:
    """
    Extract year, month, and day from a numpy datetime64 object.
    """
    if isinstance(time_point, np.datetime64):
        time_stamp = pd.Timestamp(time_point)
        return time_stamp.year, time_stamp.month, time_stamp.day
    else:
        raise ValueError("Invalid time point format.")
        return None, None, None


def insert_time_points(engine, time_points: list):
    """
    Insert time points into the database.
    """
    time_points = []
    # extract year, month, day from the time points
    for time_point in time_points:
        year, month, day = extract_time_point(time_point)
        if year is not None and month is not None and day is not None:
            time_points.append(TimePoint(year=year, month=month, day=day))

    add_data_list(engine, time_points)
    print("Time points inserted.")


def insert_var_types(engine, var_types: list):
    """
    Insert variable types into the database.
    """
    var_types = [
        VarType(
            name=var_type["name"],
            unit=var_type["unit"],
            description=var_type["description"],
        )
        for var_type in var_types
    ]
    add_data_list(engine, var_types)
    print("Variable types inserted.")


def insert_var_values(engine, var_values: list):
    """
    Insert variable values into the database.
    """
    var_values = [
        VarValue(
            grid_id=var_value["grid_id"],
            time_id=var_value["time_id"],
            var_id=var_value["var_id"],
            value=var_value["value"],
        )
        for var_value in var_values
    ]
    add_data_list(engine, var_values)
    print("Variable values inserted.")
