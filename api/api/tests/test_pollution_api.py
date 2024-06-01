import pytest
from fastapi.testclient import TestClient

from api.config import api_version
from api.main import app

API_PREFIX = f"/api/{api_version}"
client = TestClient(app)


@pytest.mark.parametrize("lat, lon", [
    (29.7756796, -95.4888013),
    (40.7277478, -74.0000374),
    (55.7545835, 37.6137138),
    (39.905245, 116.4050653)
])
def test_get_city_pollution(lat, lon):
    max_channels = {
        "r": 255,
        "g": 255,
        "b": 255,
        "a": 255
    }
    res = client.get(f"{API_PREFIX}/lp?lat={lat}&lon={lon}")
    assert res.json() == max_channels


@pytest.mark.parametrize("lat, lon", [
    (76., -74.),
    (-65., -74.)
])
def test_out_of_bounds(lat, lon):
    empty_channels = {
        "r": 0,
        "g": 0,
        "b": 0,
        "a": 255
    }
    res = client.get(f"{API_PREFIX}/lp?lat={lat}&lon={lon}")
    assert res.json() == empty_channels
