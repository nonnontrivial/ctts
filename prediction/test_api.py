from fastapi.testclient import TestClient
from .api import app, PREFIX

client = TestClient(app)


def test_get_prediction():
    lat = -30.2466
    lon = -70.7494
    r = client.get(f"/{PREFIX}/prediction?lat={lat}&lon={lon}")
    assert r.status_code == 200
