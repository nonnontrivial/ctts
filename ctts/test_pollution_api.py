from fastapi.testclient import TestClient

from .api import app
from .constants import API_PREFIX

client = TestClient(app)

def test_get_pollution_without_lat_lon():
    r = client.get(f"{API_PREFIX}/pollution")
    assert r.status_code != 200
    assert "detail" in r.json()

def test_get_pollution():
    dense_cities = {
        (29.7756796, -95.4888013),
        (40.7277478, -74.0000374),
        (55.7545835, 37.6137138),
        (39.905245, 116.4050653)
    }
    for lat,lon in dense_cities:
        r = client.get(f"{API_PREFIX}/pollution?lat={lat}&lon={lon}")
        assert r.status_code == 200
        assert r.json() == {
           	"r": 255,
           	"g": 255,
           	"b": 255,
           	"a": 255
        }

def test_get_pollution_out_of_bounds():
    out_of_bounds_coords = {(76.,-74.), (-65.,-74.)}
    for lat,lon in out_of_bounds_coords:
        r = client.get(f"{API_PREFIX}/pollution?lat={lat}&lon={lon}")
        assert r.status_code == 200
        assert r.json() == {
           	"r": 0,
           	"g": 0,
           	"b": 0,
           	"a": 255
        }
