from onehealth_db import postgresql_database as db
from fastapi import FastAPI
from contextlib import asynccontextmanager


db_url = "postgresql+psycopg2://postgres:postgres@localhost:5432/onehealth_db"


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.initialize_database(db_url, replace=False)
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {"message": "Hello World"}
