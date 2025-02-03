# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "grpcio",
#     "protobuf",
# ]
# ///

import sys
import logging
import argparse
from pathlib import Path

import grpc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

grpc_host = "localhost"
grpc_port = 50051

# we need to bring the gRPC stubs into the path
try:
    stubs_package_path = list(Path.cwd().rglob("stubs"))[0]
    sys.path.append(stubs_package_path.as_posix())
except Exception as e:
    log.error(f"failed to add stubs to path {e}")
    sys.exit(1)
else:
    import brightness_service_pb2
    import brightness_service_pb2_grpc

    log.info("loaded stubs")


def main() -> None:
    parser = argparse.ArgumentParser(description="location information")
    parser.add_argument("lat", help="latitude")
    parser.add_argument("lon", help="longitude")
    args = parser.parse_args()

    with grpc.insecure_channel(f"{grpc_host}:{grpc_port}") as channel:
        lat, lon = (float(args.lat), float(args.lon))
        stub = brightness_service_pb2_grpc.BrightnessServiceStub(channel)
        req = brightness_service_pb2.Coordinates(lat=lat, lon=lon)
        res = stub.GetBrightnessObservation(req)
        log.info(res)


if __name__ == "__main__":
    main()
