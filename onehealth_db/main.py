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
def get_cartesian(
    session: SessionDep, requested_time_point: datetime.date
) -> Union[dict, None]:
    # the frontend will request a variable over all lat, long values
    # the date input is 2016-01-01 (a date object)
    start_time = (requested_time_point.year, requested_time_point.month)
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
