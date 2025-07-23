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
    func,
)
from sqlalchemy.dialects import postgresql
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
import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Type, Tuple, List
from fastapi import HTTPException


CRS = 4326
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
        Geometry(geometry_type="POLYGON", srid=CRS)
    )


class GridPoint(Base):
    """
    Grid point table for storing latitude and longitude coordinates."""

    __tablename__ = "grid_point"

    id: Mapped[int] = mapped_column(Integer(), primary_key=True, autoincrement=True)
    latitude: Mapped[float] = mapped_column(Float())
    longitude: Mapped[float] = mapped_column(Float())

    # Geometry column for PostGIS
    point: Mapped[Geometry] = mapped_column(Geometry("POINT", srid=CRS), nullable=True)

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
        self.point = func.ST_GeomFromText(
            STR_POINT.format(str(CRS), self.longitude, self.latitude)
        )


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


class VarValueNuts(Base):
    """
    Variable value table for storing variable values at specific
    NUTS regions and time points.
    """

    __tablename__ = "var_value_nuts"

    id: Mapped[int] = mapped_column(BigInteger(), primary_key=True, autoincrement=True)
    nuts_id: Mapped[String] = mapped_column(String(), ForeignKey("nuts_def.nuts_id"))
    time_id: Mapped[int] = mapped_column(Integer(), ForeignKey("time_point.id"))
    var_id: Mapped[int] = mapped_column(Integer(), ForeignKey("var_type.id"))
    value: Mapped[float] = mapped_column(Float())

    __table_args__ = (
        UniqueConstraint("time_id", "nuts_id", "var_id", name="uq_time_nuts_var"),
        ForeignKeyConstraint(
            ["nuts_id"],
            ["nuts_def.nuts_id"],
            name="fk_nuts_id",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["time_id"],
            ["time_point.id"],
            name="fk_time_id_nuts",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["var_id"],
            ["var_type.id"],
            name="fk_var_id_nuts",
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


def insert_nuts_def(engine: engine.Engine, shapefiles_path: Path):
    """
    Insert NUTS definition data into the database.
    The shapefiles are downloaded from the Eurostat website.
    More details for downloading NUTS shapefiles can be found in
    [our data page](https://ssciwr.github.io/onehealth-db/data/#eurostats-nuts-definition)

    Five shapefiles are involved in the process:
    - `.shp`: geometry data (e.g. polygons)
    - `.shx`: shape index data
    - `.dbf`: attribute data (e.g. names, codes)
    - `.prj`: projection data (i.e. CRS)
    - `.cpg`: character encoding data

    Args:
        engine (engine.Engine): SQLAlchemy engine object.
        shapefiles_path (Path): Path to the NUTS shapefiles.
    """
    nuts_data = gpd.GeoDataFrame.from_file(shapefiles_path)
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

    # clean up the data first if nuts_def table already exists
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE nuts_def RESTART IDENTITY CASCADE"))

    # insert the data into the nuts_def table
    # here we do not use replace for if_exists because
    # the table var_value_nuts has a foreign key constraint
    # to nuts_def, so append would be safer
    nuts_data.to_postgis(NutsDef.__tablename__, engine, if_exists="append", index=False)
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
        class_type (Type[Base]): SQLAlchemy mapped class to insert data into.
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
            "point": STR_POINT.format(str(CRS), float(lon), float(lat)),
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


def get_unique_time_points(
    time_point_data: list[Tuple[np.ndarray, bool]],
) -> np.ndarray:
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
    return np.sort(unique_time_points)


def insert_time_points(
    session: Session, time_point_data: list[Tuple[np.ndarray, bool]]
):
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
    add_data_list_bulk(session, var_types, VarType)
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
) -> tuple[float, float]:
    """Insert variable values into the database.

    Args:
        engine (engine.Engine): SQLAlchemy engine object.
        ds (xr.Dataset): xarray dataset with variable data.
        var_name (str): Name of the variable to insert.
        grid_id_map (dict): Mapping of grid points to IDs.
        time_id_map (dict): Mapping of time points to IDs.
        var_id_map (dict): Mapping of variable types to IDs.
        to_monthly (bool): Whether to convert yearly data to monthly data. Defaults to False.
    Returns:
        tuple: A tuple containing the time taken to convert yearly data to monthly data,
            and the time taken to insert the variable values.
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


def get_time_points(
    session: Session,
    start_time_point: Tuple[int, int],
    end_time_point: Tuple[int, int] | None = None,
) -> List[TimePoint]:
    """Get time points from the database that fall within a specified range.

    Args:
        session (Session): SQLAlchemy session object.
        start_time_point (Tuple[int, int]): Start time point as (year, month).
        end_time_point (Tuple[int, int] | None): End time point as (year, month).
            If None, only the start time point is used.

    Returns:
        List[TimePoint]: List of TimePoint objects within the specified range.
    """
    if end_time_point is None:
        end_time_point = start_time_point

    return (
        session.query(TimePoint)
        .filter(
            (TimePoint.year > start_time_point[0])
            | (
                (TimePoint.year == start_time_point[0])
                & (TimePoint.month >= start_time_point[1])
            ),
            (TimePoint.year < end_time_point[0])
            | (
                (TimePoint.year == end_time_point[0])
                & (TimePoint.month <= end_time_point[1])
            ),
        )
        .all()
    )


def get_grid_points(
    session: Session, area: None | Tuple[float, float, float, float] = None
) -> List[GridPoint]:
    """Get grid points from the database that fall within a specified area.
    Args:
        session (Session): SQLAlchemy session object.
        area (None | Tuple[float, float, float, float]):
            Area as (North, West, South, East).
            If None, all grid points are returned.
    Returns:
        List[GridPoint]: List of GridPoint objects within the specified area.
    """
    if area is None:
        return session.query(GridPoint).all()

    north, west, south, east = area
    return (
        session.query(GridPoint)
        .filter(
            GridPoint.latitude <= north,
            GridPoint.latitude >= south,
            GridPoint.longitude >= west,
            GridPoint.longitude <= east,
        )
        .all()
    )


def get_var_types(
    session: Session,
    var_names: None | List[str] = None,
) -> List[VarType]:
    """Get variable types from the database with names specified in a list.

    Args:
        session (Session): SQLAlchemy session object.
        var_names (None | List[str]): List of variable names to filter by.
            If None, all variable types are returned.

    Returns:
        List[VarType]: List of VarType objects with the specified names.
    """
    if var_names is None:
        return session.query(VarType).all()

    return session.query(VarType).filter(VarType.name.in_(var_names)).all()


def sort_grid_points_get_ids(
    grid_points: List[GridPoint],
) -> tuple[dict, list[float], list[float]]:
    # Sort and deduplicate latitudes and longitudes
    latitudes = sorted({gp.latitude for gp in grid_points})
    longitudes = sorted({gp.longitude for gp in grid_points})

    # Create fast index maps for latitude and longitude
    lat_to_index = {lat: i for i, lat in enumerate(latitudes)}
    lon_to_index = {lon: i for i, lon in enumerate(longitudes)}

    # Map grid_id to (lat_index, lon_index)
    grid_ids = {
        gp.id: (lat_to_index[gp.latitude], lon_to_index[gp.longitude])
        for gp in grid_points
    }
    return grid_ids, latitudes, longitudes


def get_var_values_cartesian(
    session: Session,
    start_time_point: Tuple[int, int],
    end_time_point: Tuple[int, int] | None = None,
    var_names: None | List[str] = None,
) -> dict:
    """Get variable values for a cartesian map.

    Args:
        session (Session): SQLAlchemy session object.
        start_time_point (Tuple[int, int]): Start time point as (year, month).
        end_time_point (Tuple[int, int] | None): End time point as (year, month).
            If None, only the start time point is used.
        var_names (None | List[str]): List of variable names to filter by.
            If None, all variable types are used.

    Returns:
        dict: a dict with (time, latitude, longitude, var_value) keys.
    """
    # get the time points and their ids
    time_points = get_time_points(session, start_time_point, end_time_point)

    if not time_points:
        print("No time points found in the specified range.")
        raise HTTPException(
            status_code=400, detail="Missing data for requested time point."
        )

    # create a list of time points and their ids
    time_values_datetime = [
        datetime.date(year=tp.year, month=tp.month, day=1) for tp in time_points
    ]
    time_ids = {tp.id: tidx for tidx, tp in enumerate(time_points)}

    # get all the grid points and their ids
    grid_points = session.query(GridPoint).all()
    if not grid_points:
        print("No grid points found in the database.")
        raise HTTPException(
            status_code=400, detail="No grid points found in the database."
        )
    # Sort and deduplicate latitudes and longitudes
    grid_ids, latitudes, longitudes = sort_grid_points_get_ids(grid_points)

    # get variable types and their ids
    var_types = get_var_types(session, var_names)
    if not var_types:
        print("No variable types found in the specified names.")
        raise HTTPException(
            status_code=400, detail="Missing variable type for requested time point."
        )

    # get variable values for each grid point and time point
    values_list = []
    for vt in var_types:
        values = (
            session.query(VarValue)
            .filter(
                VarValue.grid_id.in_(grid_ids.keys()),
                VarValue.time_id.in_(time_ids.keys()),
                VarValue.var_id == vt.id,
            )
            .all()
        )

        values_list.append([])

        # fill the values array with the variable values
        for vv in values:
            values_list[-1].append(vv.value)

    mydict = {
        "time": time_values_datetime,
        "latitude": latitudes,
        "longitude": longitudes,
        "var_value": values_list,
        "var_names": [vt.name for vt in var_types],
        "var_units": [vt.unit for vt in var_types],
    }
    return mydict


def get_var_values_cartesian_for_download(
    session: Session,
    start_time_point: Tuple[int, int],
    end_time_point: Tuple[int, int] | None = None,
    area: None | Tuple[float, float, float, float] = None,
    var_names: None | List[str] = None,
    netcdf_file: str = "cartesian_grid_data_onehealth.nc",
) -> dict:
    """Get variable values for a cartesian map.

    Args:
        session (Session): SQLAlchemy session object.
        start_time_point (Tuple[int, int]): Start time point as (year, month).
        end_time_point (Tuple[int, int] | None): End time point as (year, month).
            If None, only the start time point is used.
        area (None | Tuple[float, float, float, float]):
            Area as (North, West, South, East).
            If None, all grid points are used.
        var_names (None | List[str]): List of variable names to filter by.
            If None, all variable types are used.
        netcdf_file (str): Name of the NetCDF file to save the dataset.

    Returns:
        dict: a dict with (time, latitude, longitude, var_value) keys.
            time or var_value is empty if no data is found.
    """
    # get the time points and their ids
    time_points = get_time_points(session, start_time_point, end_time_point)

    if not time_points:
        print("No time points found in the specified range.")
        raise HTTPException(
            status_code=400, detail="Missing data for requested time point."
        )

    # create a list of time points and their ids
    time_values = [
        np.datetime64(pd.Timestamp(year=tp.year, month=tp.month, day=1), "ns")
        for tp in time_points
    ]
    time_ids = {tp.id: tidx for tidx, tp in enumerate(time_points)}

    # get the grid points and their ids
    grid_points = get_grid_points(session, area)

    if not grid_points:
        print("No grid points found in the specified area.")
        raise HTTPException(
            status_code=400, detail="No grid points found in specified area."
        )

    # Sort and deduplicate latitudes and longitudes
    grid_ids, latitudes, longitudes = sort_grid_points_get_ids(grid_points)

    # get variable types and their ids
    var_types = get_var_types(session, var_names)
    if not var_types:
        print("No variable types found in the specified names.")
        raise HTTPException(
            status_code=400, detail="Missing variable type for requested time point."
        )

    # create an empty dataset
    ds = xr.Dataset(
        coords={
            "time": ("time", time_values),
            "latitude": ("latitude", latitudes),
            "longitude": ("longitude", longitudes),
        }
    )

    # get variable values for each grid point and time point
    for vt in var_types:
        var_name = vt.name
        values = (
            session.query(VarValue)
            .filter(
                VarValue.grid_id.in_(grid_ids.keys()),
                VarValue.time_id.in_(time_ids.keys()),
                VarValue.var_id == vt.id,
            )
            .all()
        )

        # dummy values array
        values_array = np.full(
            (len(time_values), len(latitudes), len(longitudes)), np.nan
        )

        # fill the values array with the variable values
        for vv in values:
            grid_index = grid_ids[vv.grid_id]
            lat_index, lon_index = grid_index
            time_index = time_ids[vv.time_id]
            values_array[time_index, lat_index, lon_index] = vv.value

        # add data to the dataset
        ds[var_name] = (("time", "latitude", "longitude"), values_array)

    # add variable attributes
    for var_type in var_types:
        ds[var_type.name].attrs["unit"] = var_type.unit
        ds[var_type.name].attrs["description"] = var_type.description
    # add global attributes
    ds.attrs["source"] = "OneHealth Database"
    ds.attrs["created_at"] = pd.Timestamp.now().isoformat()
    ds.attrs["description"] = "Variable values for a cartesian map from the database."

    # save the dataset to a NetCDF file
    ds.to_netcdf(netcdf_file)
    print(f"Dataset saved to {netcdf_file}")

    return {"response": "Dataset created successfully.", "netcdf_file": netcdf_file}


def get_nuts_regions(
    engine: engine.Engine,
    area: None | Tuple[float, float, float, float] = None,
) -> gpd.GeoDataFrame:
    """Get NUTS regions from the database.

    Args:
        engine (engine.Engine): SQLAlchemy engine object.
        area (None | Tuple[float, float, float, float]):
            Area as (North, West, South, East).
            If None, all NUTS regions are returned.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with NUTS region attributes and geometries.
    """
    if area is None:
        return gpd.read_postgis("SELECT * FROM nuts_def", engine, geom_col="geometry")

    north, west, south, east = area
    return gpd.read_postgis(
        f"""
        SELECT * FROM nuts_def
        WHERE ST_Intersects(
            ST_MakeEnvelope({west}, {south}, {east}, {north}, {CRS}),
            geometry
        )
        """,
        engine,
        geom_col="geometry",
    )


def get_grid_ids_in_nuts(
    engine: engine.Engine,
    nuts_regions: gpd.GeoDataFrame,
) -> List[int]:
    """Get grid point IDs that are within the NUTS regions.

    Args:
        engine (engine.Engine): SQLAlchemy engine object.
        nuts_regions (gpd.GeoDataFrame): GeoDataFrame with NUTS region geometries.

    Returns:
        List[int]: List of grid point IDs that intersect with the NUTS regions.
    """
    if nuts_regions.empty:
        return []

    sql = """
    SELECT id, point as geometry
    FROM grid_point
    """

    # turn the grid points into a GeoDataFrame
    grid_points_gdf = gpd.read_postgis(
        sql,
        engine,
        geom_col="geometry",
        crs=f"EPSG:{str(CRS)}",
    )

    # filter grid points that intersect with NUTS regions
    filtered_grid_points_gdf = gpd.sjoin(
        grid_points_gdf, nuts_regions, how="inner", predicate="intersects"
    )

    return sorted(set(filtered_grid_points_gdf["id"].tolist()))


def get_var_values_nuts(
    engine: engine.Engine,
    session: Session,
    start_time_point: Tuple[int, int],
    end_time_point: Tuple[int, int] | None = None,
    area: None | Tuple[float, float, float, float] = None,
    var_names: None | List[str] = None,
    shapefile: str | None = None,
) -> gpd.GeoDataFrame | None:
    """Get variable values for NUTS regions.

    Args:
        engine (engine.Engine): SQLAlchemy engine object.
        session (Session): SQLAlchemy session object.
        start_time_point (Tuple[int, int]): Start time point as (year, month).
        end_time_point (Tuple[int, int] | None): End time point as (year, month).
            If None, only the start time point is used.
        area (None | Tuple[float, float, float, float]):
            Area as (North, West, South, East).
            If None, all grid points are used.
        var_names (None | List[str]): List of variable names to filter by.
            If None, all variable types are used.
        netcdf_file (str | None): Path to the NetCDF file to save the dataset.
            If None, the dataset is not saved to a file.

    Returns:
        gpd.GeoDataFrame | None: GeoDataFrame with NUTS region attributes
            and variable values for each NUTS region.
            None if no data is found.
    """
    # TODO: shorten or simplify this function
    # get the time points and their ids
    time_points = get_time_points(session, start_time_point, end_time_point)

    if not time_points:
        print("No time points found in the specified range.")
        return None
    time_ids = [tp.id for tp in time_points]

    # get the nuts regions
    nuts_regions = get_nuts_regions(engine, area)
    if nuts_regions.empty:
        print("No NUTS regions found in the specified area.")
        return None

    # find grid point IDs inside the NUTS regions
    grid_ids_in_nuts = get_grid_ids_in_nuts(engine, nuts_regions)

    # get variable types and their ids
    var_types = get_var_types(session, var_names)
    if not var_types:
        print("No variable types found in the specified names.")
        return None
    var_ids = [vt.id for vt in var_types]

    # get variable values for each grid point and time point
    query = (
        session.query(
            VarValue.value.label("var_value"),
            GridPoint.point.label("geometry"),
            TimePoint.year,
            TimePoint.month,
            TimePoint.day,
            VarType.name.label("var_name"),
        )
        .join(GridPoint, VarValue.grid_id == GridPoint.id)
        .join(TimePoint, VarValue.time_id == TimePoint.id)
        .join(VarType, VarValue.var_id == VarType.id)
        .filter(
            GridPoint.id.in_(grid_ids_in_nuts),
            TimePoint.id.in_(time_ids),
            VarType.id.in_(var_ids),
        )
    )
    compiled_query = query.statement.compile(
        dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
    )
    var_values = gpd.read_postgis(
        compiled_query,
        engine,
        params={
            "grid_ids_in_nuts": list(grid_ids_in_nuts),
            "time_ids": list(time_ids),
            "var_ids": list(var_ids),
        },
        geom_col="geometry",
    )
    if var_values.empty:
        print("No variable values found for the specified criteria.")
        return None

    # convert year, month, day to np.datetime64
    var_values["time"] = pd.to_datetime(
        var_values[["year", "month", "day"]].assign(day=1)
    )
    var_values = var_values.drop(columns=["year", "month", "day"])

    # get variable values for each NUTS region
    aggregated_by_nuts = (
        gpd.sjoin(
            var_values,
            nuts_regions,
            how="inner",
            predicate="intersects",
        )
        .groupby(["nuts_id", "var_name", "time"])
        .agg(
            {
                "var_value": "mean",  # average value for the variable
            }
        )
        .reset_index()
    )
    # merge the aggregated values with the NUTS regions
    nuts_var_values = nuts_regions.merge(aggregated_by_nuts, on="nuts_id")

    # save to shapefile if specified
    if shapefile:
        nuts_var_values.to_file(shapefile, driver="ESRI Shapefile")
        print(f"NUTS variable values saved to {shapefile}")

    return nuts_var_values


def insert_var_value_nuts(
    engine: engine.Engine,
    ds: xr.Dataset,
    time_id_map: dict,
    var_name: str,
    var_id_map: dict,
) -> float:
    """Insert variable values for NUTS regions into the database.

    Args:
        engine (engine.Engine): SQLAlchemy engine object.
        ds (xr.Dataset): xarray dataset with dimensions (time, nuts_id).
        time_id_map (dict): Mapping of time points to IDs.
        var_name (str): Name of the variable to insert.
        var_id_map (dict): Mapping of variable names to variable type IDs.

    Returns:
        float: The time taken to insert the variable values.
    """
    # get the variable id
    var_id = var_id_map.get(var_name)
    if var_id is None:
        raise ValueError(f"Variable {var_name} not found in var_type table.")

    # values of the variable
    var_data = (
        ds[var_name].dropna(dim="nuts_id", how="all").load()
    )  # load data into memory

    # using stack() from xarray to vectorize the data
    stacked_var_data = var_data.stack(points=("time", "nuts_id"))
    stacked_var_data = stacked_var_data.dropna("points")

    # get values of each dim
    time_vals = stacked_var_data["time"].values.astype("datetime64[ns]")
    nuts_ids = stacked_var_data["nuts_id"].values

    # create vectorized mapping
    # normalize time before mapping as the time in isimip is 12:00:00
    # TODO: find an optimal way to do this
    get_time_id = np.vectorize(
        lambda t: time_id_map.get(np.datetime64(pd.Timestamp(t).normalize(), "ns"))
    )

    # TODO: work still in progress
    return time_vals, nuts_ids, get_time_id
