from fastapi.testclient import TestClient

from .api import app
from .constants import API_PREFIX

client = TestClient(app)

def test_get_pollution():
    cities = {(29.7756796, -95.4888013), (40.7277478, -74.0000374)}
    for lat,lon in cities:
        r = client.get(f"{API_PREFIX}/pollution?lat={lat}&lon={lon}")
        assert r.status_code == 200
        assert r.json() == {
           	"r": 255,
           	"g": 255,
           	"b": 255,
           	"a": 255
        }
