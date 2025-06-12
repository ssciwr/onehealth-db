from onehealth_db import postgresql_database as db
from fastapi import FastAPI, Depends
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session
from typing import Annotated
import time


db_url = "postgresql+psycopg2://postgres:postgres@localhost:5432/onehealth_db"


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.initialize_database(db_url, replace=False)
    yield


def get_session(engine):
    session = db.create_session(engine)
    yield session


SessionDep = Annotated[Session, Depends(get_session)]


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/cartesian/")
def get_cartesian(session: SessionDep) -> float | int | str | None:
    latitude = -6.25
    longitude = 106.75
    year = 2021
    month = 1
    day = 1
    var_name = "t2m"
    t_start_retrieving = time.time()
    var_value = db.get_var_value(
        session, var_name, latitude, longitude, year, month, day
    )
    t_end_retrieving = time.time()
    print(
        f"Retrieved {var_name} value: {var_value} in \
        {t_end_retrieving - t_start_retrieving} seconds."
    )
    return var_value
