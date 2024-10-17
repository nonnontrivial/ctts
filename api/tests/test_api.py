from concurrent import futures
from unittest.mock import patch

import pytest
import grpc

from api.stubs import brightness_service_pb2_grpc
from api.stubs import brightness_service_pb2
from api.service.brightness_servicer import BrightnessServicer


@pytest.fixture(scope='module')
def grpc_server():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    brightness_service_pb2_grpc.add_BrightnessServiceServicer_to_server(BrightnessServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()

    yield
    server.stop(0)

@pytest.fixture(scope='module')
def brightness_service_stub(grpc_server):
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = brightness_service_pb2_grpc.BrightnessServiceStub(channel)
        yield stub

def test_apis(brightness_service_stub):
    request = brightness_service_pb2.Coordinates(lat=42.,lon=42.)

    response = brightness_service_stub.GetPollution(request)
    assert isinstance(response.r,int)
    assert isinstance(response.g,int)
    assert isinstance(response.b,int)
    assert isinstance(response.a,int)

@patch('api.service.open_meteo.open_meteo_client.requests.get')
def test_brightness_observation(mock_get, brightness_service_stub):
    from datetime import datetime,UTC

    mock_response = mock_get.return_value
    mock_response.status_code = 200
    mock_response.json.return_value={"elevation":0.,"hourly":{"cloud_cover":[0]*24}}

    lat,lon = (42.,42.)
    request = brightness_service_pb2.Coordinates(lat=lat,lon=lon)

    response = brightness_service_stub.GetBrightnessObservation(request)

    assert response.lat == lat
    assert response.lon == lon
    parsed_utc_iso = datetime.fromisoformat(response.utc_iso)
    assert parsed_utc_iso.date() == datetime.now(UTC).date()
    assert response.mpsas > 0 and response.mpsas < 22
