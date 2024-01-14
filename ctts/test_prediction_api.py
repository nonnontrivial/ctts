from fastapi.testclient import TestClient

from .api import app
from .constants import API_PREFIX

client = TestClient(app)
lat, lon = (-30.2466, -70.7494)


def test_get_prediction_bad_status_without_lat_lon():
    r = client.get(f"{API_PREFIX}/prediction")
    assert r.status_code != 200


def test_get_prediction_bad_status_with_bad_astro_twilight():
    astro_twilight_type = "bad"
    r = client.get(
        f"{API_PREFIX}/prediction?lat={lat}&lon={lon}&astro_twilight_type={astro_twilight_type}"
    )
    assert r.status_code == 422


def test_get_prediction():
    astro_twilight_type = "next"
    r = client.get(
        f"{API_PREFIX}/prediction?lat={lat}&lon={lon}&astro_twilight_type={astro_twilight_type}"
    )
    assert r.status_code == 200
    assert list(r.json().keys()) == ["sky_brightness", "astronomical_twilight_iso"]
