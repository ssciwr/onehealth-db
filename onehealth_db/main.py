from onehealth_db import postgresql_database as db
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, text
from typing import Annotated, Union
import datetime
import dotenv
import os
import logging
import ipaddress

logging.basicConfig(level=logging.DEBUG)

# get the db url from dotenv
dotenv.load_dotenv()
db_url = os.environ.get("DB_URL")
# with the initialization of the engine like this,
# we cannot use the same db mocking strategy as in the other tests
# because the engine is created before the tests are run.
if not db_url:
    raise ValueError("DB_URL environment variable is not set.")
engine = create_engine(db_url)

# get the IP address from the environment variable
ip_address = os.environ.get("IP_ADDRESS")
# check that the IP address is a string
if not isinstance(ip_address, str):
    raise ValueError("IP_ADDRESS environment variable must be a string.")
try:
    ipaddress.IPv4Address(ip_address)
except Exception:
    raise ValueError(
        f"IP_ADDRESS environment variable is not a valid IPv4 address: {ip_address}"
    )
allowed_origins = [
    f"http://{ip_address}",
    "http://localhost",
    "http://127.0.0.1",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.initialize_database(db_url, replace=False)
    yield


def get_session():
    if engine is None:
        raise RuntimeError(
            "Database engine is not initialized. \
                           Please check the DB_URL environment variable."
        )
    session = db.create_session(engine)
    try:
        yield session
    finally:
        session.close()


SessionDep = Annotated[Session, Depends(get_session)]


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    # update this to https later when using ssl
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    session: SessionDep,
    requested_time_point: datetime.date,
    requested_variable_value: str | None,
) -> Union[dict, None]:
    # the frontend will request a variable over all available lat, long values for that variable
    # the date input is 2016-01-01 (a date object)
    # the variable input is a matching string, ie "t2m" for temperature
    # the variable name will be supplied via the model yaml files for each selected model
    if not isinstance(requested_time_point, datetime.date):
        return {"error": "Invalid date format. Use YYYY-MM-DD."}
    date_requested = (requested_time_point.year, requested_time_point.month)
    var_name = requested_variable_value
    try:
        var_value = db.get_var_values_cartesian(
            session,
            time_point=date_requested,
            var_name=var_name,
        )
        return {"result": var_value}
    except Exception as e:
        return {"error": str(e)}


@app.get("/nuts_data")
def get_nuts_data(
    session: SessionDep,
    requested_time_point: datetime.date,
    requested_variable_value: str | None,
) -> Union[dict, None]:
    # the frontend will request a variable over all available lat, long values for that variable
    # the date input is 2016-01-01 (a date object)
    # the variable input is a matching string, ie "t2m" for temperature
    # the variable name will be supplied via the model yaml files for each selected model
    if not isinstance(requested_time_point, datetime.date):
        return {"error": "Invalid date format. Use YYYY-MM-DD."}
    date_requested = (requested_time_point.year, requested_time_point.month)
    var_name = requested_variable_value
    try:
        var_value = db.get_var_values_nuts(
            session,
            time_point=date_requested,
            var_name=var_name,
        )
        return {"result": var_value}
    except Exception as e:
        return {"error": str(e)}
