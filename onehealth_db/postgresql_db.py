from sqlalchemy import create_engine, Column, Float, TIMESTAMP, Index, text
from sqlalchemy.ext.declarative import declarative_base
from geoalchemy2 import Geometry
from sqlalchemy.orm import sessionmaker


# this work is in progress, tests are still needed!
# Base declarative class
Base = declarative_base()


class GeoGridStats(Base):
    __tablename__ = "geo_grid_stats"

    time = Column(TIMESTAMP, primary_key=True)
    latitude = Column(Float, primary_key=True)
    longitude = Column(Float, primary_key=True)
    t2m = Column(Float)
    tp = Column(Float)
    popu = Column(Float)

    # indices
    __table_args__ = (
        Index("idx_time", "time"),
        Index("idx_spatial", "latitude", "longitude", postgresql_using="gist"),
    )


# create engine and connect to PostgreSQL database
engine = create_engine(
    "postgresql+psycopg2://onehealth_admin:onehealth123@localhost:5432/onehealth_db"
)

# install PostGIS extension
with engine.connect() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))

# create table
Base.metadata.create_all(engine)
