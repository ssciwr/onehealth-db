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
    engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.exc import SQLAlchemyError
from geoalchemy2 import Geometry, WKBElement
from sqlalchemy.orm.session import sessionmaker, Session
import geopandas as gpd
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Type


STR_CRS = "4326"
STR_POINT = "SRID={};POINT({} {})"
BATCH_SIZE = 10000
MAX_WORKERS = 4


# Base declarative class
class Base(DeclarativeBase):
    """
    Base class for all models in the database."""

    pass


class NutsDef(Base):
    """
    NUTS definition table."""

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
    """
    Grid point table for storing latitude and longitude coordinates."""

    __tablename__ = "grid_point"

    id: Mapped[int] = mapped_column(Integer(), primary_key=True, autoincrement=True)
    latitude: Mapped[float] = mapped_column(Float())
    longitude: Mapped[float] = mapped_column(Float())

    # Geometry column for PostGIS
    point: Mapped[Geometry] = mapped_column(Geometry("POINT", srid=4326), nullable=True)

    __table_args__ = (
        Index("idx_point_gridpoint", "point", postgresql_using="gist"),
        UniqueConstraint("latitude", "longitude", name="uq_lat_lon"),
    )

    def __init__(self, latitude, longitude, **kw):
        super().__init__(**kw)
        self.latitude = latitude
        self.longitude = longitude
        # add value of point automatically,
        # only works when using the constructor, i.e. session.add()
        self.point = STR_POINT.format(STR_CRS, self.longitude, self.latitude)


class TimePoint(Base):
    """
    Time point table for storing year, month, and day."""

    __tablename__ = "time_point"

    id: Mapped[int] = mapped_column(Integer(), primary_key=True, autoincrement=True)
    year: Mapped[int] = mapped_column(Integer())
    month: Mapped[int] = mapped_column(Integer())
    day: Mapped[int] = mapped_column(Integer())

    __table_args__ = (
        UniqueConstraint("year", "month", "day", name="uq_year_month_day"),
    )


class VarType(Base):
    """
    Variable type table for storing variable metadata."""

    __tablename__ = "var_type"

    id: Mapped[int] = mapped_column(Integer(), primary_key=True, autoincrement=True)
    name: Mapped[String] = mapped_column(String())
    unit: Mapped[String] = mapped_column(String())
    description: Mapped[String] = mapped_column(String(), nullable=True)

    __table_args__ = (UniqueConstraint("name", name="uq_var_name"),)


class VarValue(Base):
    """
    Variable value table for storing variable values at specific
    grid points and time points.
    """

    __tablename__ = "var_value"

    id: Mapped[int] = mapped_column(BigInteger(), primary_key=True, autoincrement=True)
    grid_id: Mapped[int] = mapped_column(Integer(), ForeignKey("grid_point.id"))
    time_id: Mapped[int] = mapped_column(Integer(), ForeignKey("time_point.id"))
    var_id: Mapped[int] = mapped_column(Integer(), ForeignKey("var_type.id"))
    value: Mapped[float] = mapped_column(Float())

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


def install_postgis(engine: engine.Engine):
    """
    Install PostGIS extension on the database.

    Args:
        engine (engine.Engine): SQLAlchemy engine object.
    """
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
        print("PostGIS extension installed.")


def create_session(engine: engine.Engine) -> Session:
    """
    Create a new session for the database.

    Args:
        engine (engine.Engine): SQLAlchemy engine object.

    Returns:
        Session: SQLAlchemy session object.
    """
    session_class = sessionmaker(bind=engine)
    return session_class()


def create_tables(engine: engine.Engine):
    """
    Create all tables in the database.

    Args:
        engine (engine.Engine): SQLAlchemy engine object.
    """
    Base.metadata.create_all(engine)
    print("All tables created.")


def create_or_replace_tables(engine: engine.Engine):
    """
    Create or replace tables in the database.

    Args:
        engine (engine.Engine): SQLAlchemy engine object.
    """
    Base.metadata.drop_all(engine)
    print("All tables dropped.")
    create_tables(engine)


def initialize_database(db_url: str, replace: bool = False):
    """
    Initialize the database by creating the engine and tables, and installing PostGIS.
    If replace is True, it will drop and recreate the tables.

    Args:
        db_url (str): Database URL for SQLAlchemy.
        replace (bool): Whether to drop and recreate the tables. Defaults to False.
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


def insert_nuts_def(engine: engine.Engine, shapefile_path: Path):
    """
    Insert NUTS definition data into the database.

    Args:
        engine (engine.Engine): SQLAlchemy engine object.
        shapefile_path (Path): Path to the NUTS shapefile.
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
    print("NUTS definition data inserted.")


def add_data_list(session: Session, data_list: list):
    """
    Add a list of data instances to the database.

    Args:
        session (Session): SQLAlchemy session object.
        data_list (list): List of data instances to add.
    """
    try:
        session.add_all(data_list)
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        print(f"Error inserting data: {e}")


def add_data_list_bulk(session: Session, data_dict_list: list, class_type: Type[Base]):
    """
    Add a list of data to the database in bulk.

    Args:
        session (Session): SQLAlchemy session object.
        data_dict_list (list): List of dictionaries containing data to insert.
        class_type (Type[Base]): SQLAlchemy model class type to insert data into.
    """
    try:
        session.bulk_insert_mappings(class_type, data_dict_list)
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        print(f"Error inserting data: {e}")


def insert_grid_points(session: Session, latitudes: np.ndarray, longitudes: np.ndarray):
    """
    Insert grid points into the database.

    Args:
        session (Session): SQLAlchemy session object.
        latitudes (np.ndarray): Array of latitudes.
        longitudes (np.ndarray): Array of longitudes.
    """
    # create list of dictionaries for bulk insert
    grid_points = [
        {
            "latitude": float(lat),
            "longitude": float(lon),
            "point": STR_POINT.format(STR_CRS, float(lon), float(lat)),
        }
        for lat in latitudes
        for lon in longitudes
    ]
    add_data_list_bulk(session, grid_points, GridPoint)
    print("Grid points inserted.")


def extract_time_point(
    time_point: np.datetime64,
) -> tuple[int, int, int, int, int, int]:
    """
    Extract year, month, and day from a numpy datetime64 object.

    Args:
        time_point (np.datetime64): Numpy datetime64 object representing a time point.

    Returns:
        tuple: A tuple containing year, month, day, hour, minute, second.
    """
    if isinstance(time_point, np.datetime64):
        time_stamp = pd.Timestamp(time_point)
        return (
            time_stamp.year,
            time_stamp.month,
            time_stamp.day,
            time_stamp.hour,
            time_stamp.minute,
            time_stamp.second,
        )
    else:
        raise ValueError("Invalid time point format.")


def get_unique_time_points(time_point_data: list[(np.ndarray, bool)]) -> np.ndarray:
    """Get the unique of time points.

    Args:
        time_point_data: List of tuples containing time point data, and the yearly flag.
            If flag is True, the time point needs to be converted to monthly.

    Returns:
        np.ndarray: Unique of (sorted) time points as a numpy array.
    """
    time_points = []
    for tpd, yearly in time_point_data:
        if not yearly:
            # assume it's monthly TODO
            time_points.append(tpd)
        else:
            # convert to monthly for the whole range
            if np.datetime64(tpd[0]) > np.datetime64(tpd[-1]):
                # sort before converting
                tpd = np.sort(tpd)

            start_of_year = pd.Timestamp(
                year=extract_time_point(np.datetime64(tpd[0]))[0], month=1, day=1
            )
            end_of_year = pd.Timestamp(
                year=extract_time_point(np.datetime64(tpd[-1]))[0], month=12, day=1
            )
            time_points.append(
                pd.date_range(start=start_of_year, end=end_of_year, freq="MS").values
            )

    if not time_points:
        return np.array([])

    concatenated = np.concatenate(time_points)
    unique_time_points = np.unique(concatenated)
    return sorted(unique_time_points)


def insert_time_points(session: Session, time_point_data: list[(np.ndarray, bool)]):
    """Insert time points into the database.

    Args:
        session (Session): SQLAlchemy session object.
        time_point_data (list[(np.ndarray, bool)]): List of tuples containing
            time point data, and its flag.
            If flag is True, the time point needs to be converted to monthly.
    """
    time_point_values = []
    # get the overlap of the time points
    time_points = get_unique_time_points(time_point_data)

    # extract year, month, day from the time points
    for time_point in time_points:
        year, month, day, _, _, _ = extract_time_point(time_point)
        if year is not None and month is not None and day is not None:
            time_point_values.append(
                {
                    "year": year,
                    "month": month,
                    "day": day,
                }
            )

    add_data_list_bulk(session, time_point_values, TimePoint)
    print("Time points inserted.")


def insert_var_types(session: Session, var_types: list[dict]):
    """
    Insert variable types into the database.

    Args:
        session (Session): SQLAlchemy session object.
        var_types (list[dict]): List of dictionaries containing variable type data.
    """
    var_types = [
        VarType(
            name=var_type["name"],
            unit=var_type["unit"],
            description=var_type["description"],
        )
        for var_type in var_types
    ]
    add_data_list(session, var_types)
    print("Variable types inserted.")


def get_id_maps(session: Session) -> tuple[dict, dict, dict]:
    """
    Get ID maps for grid points, time points, and variable types.

    Args:
        session (Session): SQLAlchemy session object.

    Returns:
        tuple: A tuple containing three dictionaries:\n
            - grid_id_map: Mapping of (latitude, longitude) to grid point ID.\n
            - time_id_map: Mapping of datetime64 to time point ID.\n
            - var_id_map: Mapping of variable name to variable type ID.
    """
    grid_points = session.query(
        GridPoint.id, GridPoint.latitude, GridPoint.longitude
    ).all()
    grid_id_map = {(lat, lon): grid_id for grid_id, lat, lon in grid_points}

    time_id_map = {
        np.datetime64(pd.to_datetime(f"{row.year}-{row.month}-{row.day}"), "ns"): row.id
        for row in session.query(TimePoint).all()
    }

    var_id_map = {row.name: row.id for row in session.query(VarType).all()}

    session.close()

    return grid_id_map, time_id_map, var_id_map


def convert_yearly_to_monthly(ds: xr.Dataset) -> xr.Dataset:
    """Convert yearly data to monthly data.

    Args:
        ds (xr.Dataset): xarray dataset with yearly data.

    Returns:
        xr.Dataset: xarray dataset with monthly data.
    """
    if ds.time.values[0] > ds.time.values[-1]:
        # sort the time points
        ds = ds.sortby("time")

    # create monthly time points
    s_y, s_m, _, s_h, s_mi, s_s = extract_time_point(ds.time.values[0])
    e_y, _, _, e_h, e_mi, e_s = extract_time_point(ds.time.values[-1])
    new_time_points = pd.date_range(
        start=pd.Timestamp(
            year=s_y, month=s_m, day=1, hour=s_h, minute=s_mi, second=s_s
        ),
        end=pd.Timestamp(year=e_y, month=12, day=1, hour=e_h, minute=e_mi, second=e_s),
        freq="MS",
    )

    # reindex dataset with new time points
    return ds.reindex(time=new_time_points, method="ffill")


def insert_var_values(
    engine: engine.Engine,
    ds: xr.Dataset,
    var_name: str,
    grid_id_map: dict,
    time_id_map: dict,
    var_id_map: dict,
    to_monthly: bool = False,
):
    """Insert variable values into the database.

    Args:
        engine (engine.Engine): SQLAlchemy engine object.
        ds (xr.Dataset): xarray dataset with variable data.
        var_name (str): Name of the variable to insert.
        grid_id_map (dict): Mapping of grid points to IDs.
        time_id_map (dict): Mapping of time points to IDs.
        var_id_map (dict): Mapping of variable types to IDs.
        to_monthly (bool): Whether to convert yearly data to monthly data.
    """
    if to_monthly:
        # convert yearly data to monthly data
        print(f"Converting {var_name} data from yearly to monthly...")
        ds = convert_yearly_to_monthly(ds)
    t_yearly_to_monthly = time.time()

    print(f"Prepare inserting {var_name} values...")
    # values of the variable
    var_data = ds[var_name]
    var_data = var_data.dropna(
        dim="latitude", how="all"
    ).load()  # load data into memory

    # get the variable id
    var_id = var_id_map.get(var_name)
    if var_id is None:
        raise ValueError(f"Variable {var_name} not found in var_type table.")

    # using stack() from xarray to vectorize the data
    stacked_var_data = var_data.stack(points=("time", "latitude", "longitude"))
    stacked_var_data = stacked_var_data.dropna("points")

    # get id for each dim
    time_vals = stacked_var_data["time"].values.astype("datetime64[ns]")
    lat_vals = stacked_var_data["latitude"].values
    lon_vals = stacked_var_data["longitude"].values

    # create vectorized mapping
    # normalize time before mapping as the time in isimip is 12:00:00
    # TODO: find an optimal way to do this
    get_time_id = np.vectorize(
        lambda t: time_id_map.get(np.datetime64(pd.Timestamp(t).normalize(), "ns"))
    )
    get_grid_id = np.vectorize(lambda lat, lon: grid_id_map.get((lat, lon)))

    time_ids = get_time_id(time_vals)
    grid_ids = get_grid_id(lat_vals, lon_vals)
    values = stacked_var_data.values.astype(float)

    # create a mask for valid values
    masks = ~np.isnan(values)

    # create bulk data for insertion
    var_values = [
        {
            "grid_id": int(grid_id),
            "time_id": int(time_id),
            "var_id": int(var_id),
            "value": float(value),
        }
        for grid_id, time_id, value, mask in zip(grid_ids, time_ids, values, masks)
        if mask and (grid_id is not None) and (time_id is not None)
    ]

    def insert_batch(batch):
        """Insert a batch of data into the database."""
        # create a new session for each batch
        session = create_session(engine)
        add_data_list_bulk(session, batch, VarValue)
        session.close()

    print(f"Start inserting {var_name} values in parallel...")
    t_start_insert = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for i in range(0, len(var_values), BATCH_SIZE):
            e_batch = i + BATCH_SIZE
            batch = var_values[i:e_batch]
            futures.append(executor.submit(insert_batch, batch))

        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass

    print(f"Values of {var_name} inserted.")
    return t_yearly_to_monthly, t_start_insert


def get_var_value(
    session: Session,
    var_name: str,
    lat: float,
    lon: float,
    year: int,
    month: int,
    day: int,
) -> float | int | str | None:
    """Get variable value from the database.

    Args:
        session (Session): SQLAlchemy session object.
        var_name (str): Name of the variable to retrieve.
        lat (float): Latitude of the grid point.
        lon (float): Longitude of the grid point.
        year (int): Year of the time point.
        month (int): Month of the time point.
        day (int): Day of the time point.

    Returns:
        float | int | str | None: Value of the variable at
            the specified grid point and time point.
    """
    if day != 1:
        print(
            "The current database only supports monthly data."
            "Retieving data for the first day of the month..."
        )
        day = 1

    result = (
        session.query(VarValue)
        .join(GridPoint, VarValue.grid_id == GridPoint.id)
        .join(TimePoint, VarValue.time_id == TimePoint.id)
        .join(VarType, VarValue.var_id == VarType.id)
        .filter(
            GridPoint.latitude == lat,
            GridPoint.longitude == lon,
            TimePoint.year == year,
            TimePoint.month == month,
            TimePoint.day == day,
            VarType.name == var_name,
        )
        .first()
    )
    return result.value if result else None
