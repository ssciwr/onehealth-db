from fastapi.testclient import TestClient
from onehealth_db.main import app


client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}


def test_db_status():
    response = client.get("/db-status")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_get_cartesian():
    # at the moment we cannot test this easily without the deployment set up
    pass
