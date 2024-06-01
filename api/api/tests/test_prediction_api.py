import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.config import api_version

client = TestClient(app)
API_PREFIX = f"/api/{api_version}"


def test_get_prediction_bad_status_without_lat_lon():
    r = client.get(f"{API_PREFIX}/predict")
    assert r.status_code != 200


@pytest.mark.parametrize("coords, lowerbound, upperbound", [
    ((-30.2466, -70.7494), 6, 25),
    ((19.8264, -155.4750), 6, 28)
])
def test_prediction(coords, lowerbound, upperbound):
    lat, lon = coords
    response = client.get(f"{API_PREFIX}/predict?lat={lat}&lon={lon}")
    assert response.status_code == 200
    brightness = response.json()["sky_brightness"]
    assert lowerbound <= brightness <= upperbound
