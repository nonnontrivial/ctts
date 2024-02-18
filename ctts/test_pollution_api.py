from fastapi.testclient import TestClient
import pytest

from .api import app
from .constants import API_PREFIX

client = TestClient(app)


max_channel_value = 255
keys = ["r", "g", "b", "a"]
max_value = {x: max_channel_value for x in keys}


@pytest.mark.parametrize(
    "lat,lon,value",
    [
        (29.7756796, -95.4888013, max_value),
        (40.7277478, -74.0000374, max_value),
        (55.7545835, 37.6137138, max_value),
        (39.905245, 116.4050653, max_value),
        (35.6895, 139.6917, max_value),
        (28.6139, 77.2090, max_value),
        (31.2304, 121.4737, max_value),
        (-23.5505, -46.6333, max_value),
    ],
)
def test_get_pollution(lat, lon, value):
    r = client.get(f"{API_PREFIX}/pollution?lat={lat}&lon={lon}")
    assert r.status_code == 200
    assert r.json() == value


def test_get_pollution_out_of_bounds():
    out_of_bounds_coords = {(76.0, -74.0), (-65.0, -74.0)}
    for lat, lon in out_of_bounds_coords:
        r = client.get(f"{API_PREFIX}/pollution?lat={lat}&lon={lon}")
        assert r.status_code == 200
        assert r.json() == {"r": 0, "g": 0, "b": 0, "a": 255}


def test_get_pollution_without_lat_lon():
    r = client.get(f"{API_PREFIX}/pollution")
    assert r.status_code != 200
    assert "detail" in r.json()
