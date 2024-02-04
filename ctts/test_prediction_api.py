from fastapi.testclient import TestClient
import pytest

from .api import app
from .constants import API_PREFIX

client = TestClient(app)


def test_get_prediction_bad_status_without_lat_lon():
    r = client.get(f"{API_PREFIX}/prediction")
    assert "detail" in r.json()
    assert r.status_code != 200


def test_get_prediction_bad_status_with_bad_astro_twilight():
    astro_twilight_type = "bad"
    lat, lon = (-30.2466, -70.7494)
    r = client.get(
        f"{API_PREFIX}/prediction?lat={lat}&lon={lon}&astro_twilight_type={astro_twilight_type}"
    )
    assert "detail" in r.json()
    assert r.status_code == 422


def test_get_prediction():
    astro_twilight_type = "nearest"
    lat, lon = (-30.2466, -70.7494)
    r = client.get(
        f"{API_PREFIX}/prediction?lat={lat}&lon={lon}&astro_twilight_type={astro_twilight_type}"
    )
    res_json = r.json()
    assert res_json["nat"] is not None
    assert res_json["nsb"] is not None
    assert r.status_code == 200
