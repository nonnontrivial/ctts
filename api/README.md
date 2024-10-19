# api

gRPC apis for [sky brightness](https://en.wikipedia.org/wiki/Sky_brightness).

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

## model

### building and training

The api depends on a model being trained from csv data.

The following commands will generate a new `model.pth` (i.e. the learned parameters):

- `python -m api.model.build` to write the csv that the model trains on
- `python -m api.model.train` to train on the data in the csv
