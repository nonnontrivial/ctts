from fastapi.testclient import TestClient

from .api import app
from .constants import API_PREFIX

client = TestClient(app)


def test_get_prediction_bad_status_without_lat_lon():
    r = client.get(f"{API_PREFIX}/prediction")
    assert r.status_code != 200


def test_get_prediction():
    lat = -30.2466
    lon = -70.7494
    r = client.get(f"{API_PREFIX}/prediction?lat={lat}&lon={lon}")
    assert r.status_code == 200
