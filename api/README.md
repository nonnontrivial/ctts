# api

api server for sky brightness

## running the tests

```sh
python3 -m pytest
```

## gRPC client example

- install dependencies `pip install grpcio-tools grpcio protobuf`
- generate the stubs using `python3 -m grpc_tools.protoc` in terms of the `.proto` file in this repo

```py
"""client which will get predicted sky brightness for the current time at
given coordinates"""
import grpc

from stubs import brightness_service_pb2
from stubs import brightness_service_pb2_grpc

host = "localhost"
port = 50051

with grpc.insecure_channel(f"{host}:{port}") as channel:
    lat,lon=(0.,0.)

    stub = brightness_service_pb2_grpc.BrightnessServiceStub(channel)

    request = brightness_service_pb2.Coordinates(lat=lat, lon=lon)
    response = stub.GetBrightnessObservation(request)

    print(response)

```
