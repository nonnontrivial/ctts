from concurrent import futures

import pytest
import grpc

from api.stubs import brightness_service_pb2_grpc
from api.stubs import brightness_service_pb2
from api.service.brightness_servicer import BrightnessServicer


@pytest.fixture(scope='module')
def grpc_server():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    # example_pb2_grpc.add_GreeterServicer_to_server(Greeter(), server)
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
    request= brightness_service_pb2.Coordinates(lat=42.,lon=42.)

    response=brightness_service_stub.GetPollution(request)
    assert isinstance(response.r,int)
    assert isinstance(response.g,int)
    assert isinstance(response.b,int)
    assert isinstance(response.a,int)

@pytest.mark.skip
def test_brightness_observation(brightness_service_stub):
    request= brightness_service_pb2.Coordinates(lat=42.,lon=42.)

    response=brightness_service_stub.GetBrightnessObservation(request)
    assert response is not None
