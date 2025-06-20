from onehealth_db import postgresql_database as db
from fastapi import FastAPI, Depends
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, text
from typing import Annotated, Union
import time
import datetime

db_url = "postgresql+psycopg2://postgres:postgres@localhost:5432/onehealth_db"
engine = create_engine(db_url)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.initialize_database(db_url, replace=False)
    yield


def get_session():
    session = db.create_session(engine)
    try:
        yield session
    finally:
        session.close()


SessionDep = Annotated[Session, Depends(get_session)]


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root(message: str = "Hello World") -> dict:
    return {"message": message}


@app.get("/db-status")
def db_status():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.get("/cartesian")
def get_cartesian(session: SessionDep) -> Union[dict, None]:
    year = 2016
    month = 1
    day = 1
    requested_time = datetime.datetime(year, month, day)
    # the frontend will request a variable over all lat, long values
    # given a datetime object, so we need to convert this to a time that
    # the database can understand
    start_time = (requested_time.year, requested_time.month)
    var_name = "t2m"
    t_start_retrieving = time.time()
    try:
        var_value = db.get_var_values_cartesian(
            session,
            start_time_point=start_time,
            end_time_point=start_time,
            area=None,
            var_names=[var_name],
            netcdf_file=None,
        )
        t_end_retrieving = time.time()
        print(
            f"Retrieved {var_name} value in \
            {t_end_retrieving - t_start_retrieving} seconds."
        )
        return {"result": var_value}
    except Exception as e:
        return {"error": str(e)}
