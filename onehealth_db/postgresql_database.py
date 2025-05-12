from sqlalchemy import create_engine, Float, TIMESTAMP, String, Integer, Index, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from geoalchemy2 import Geometry, WKBElement
from sqlalchemy.orm import sessionmaker
import geopandas as gpd
from pathlib import Path
import psycopg2


# Base declarative class
class Base(DeclarativeBase):
    pass


class GeoGridStats(Base):
    __tablename__ = "geo_grid_stats"

    time: Mapped[TIMESTAMP] = mapped_column(TIMESTAMP(), primary_key=True)
    latitude: Mapped[Float] = mapped_column(Float(), primary_key=True)
    longitude: Mapped[Float] = mapped_column(Float(), primary_key=True)
    t2m: Mapped[Float] = mapped_column(Float(), nullable=True)
    tp: Mapped[Float] = mapped_column(Float(), nullable=True)
    popu: Mapped[Float] = mapped_column(Float(), nullable=True)

    # daylight column
    daylight: Mapped[Float] = mapped_column(Float(), nullable=True)
    # Geometry column for PostGIS
    point: Mapped[Geometry] = mapped_column(Geometry("POINT", srid=4326))

    # indices
    __table_args__ = (
        Index("idx_time", "time"),
        Index("idx_spatial", "point", postgresql_using="gist"),
    )

    def __init__(self, time, latitude, longitude, t2m, tp, popu, **kw):
        super().__init__(**kw)
        self.time = time
        self.latitude = latitude
        self.longitude = longitude
        self.t2m = t2m
        self.tp = tp
        self.popu = popu

        self.point = f"SRID=4326;POINT({self.longitude} {self.latitude})"
        self.daylight = self.calculate_daylight()

    def calculate_daylight(self):
        """
        Calculate daylight hours based on latitude and time.
        This is a placeholder for the actual daylight calculation logic.
        """
        # Placeholder logic for daylight calculation
        # Replace with actual calculation as needed
        return 12.0


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


def initialize_database(db_url: str):
    """
    Initialize the database by creating the engine, session, and tables.
    """
    # create engine
    engine = create_engine(db_url, echo=True)

    # install PostGIS extension
    install_postgis(engine)

    # create tables
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
    nuts_data.to_sql(
        NutsDef.__tablename__,
        engine,
        if_exists="replace",
        index=False,
        dtype={
            "geometry": Geometry("POLYGON", srid=4326),
        },
    )


if __name__ == "__main__":
    # PostgreSQL database URL
    db_url = "postgresql+psycopg2://postgres:postgres@localhost:5432/postgres"

    # initialize the database
    engine = initialize_database(db_url)

    # path to the shapefile
    pkg_pth = Path(__file__).parent.parent
    shapefile_path = pkg_pth / "data" / "in" / "NUTS_RG_20M_2024_4326.shp"
    # insert NUTS definition data
    insert_nuts_def(engine, shapefile_path)
    # conn = psycopg2.connect(
    #     dbname="postgres",
    #     user="postgres",
    #     host="localhost",
    #     password="postgres",
    #     port="5432",
    # )
    # print("Connected to the database.")
