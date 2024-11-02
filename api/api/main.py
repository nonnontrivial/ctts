import logging
from concurrent import futures

import grpc

from .config import service_port
from .stubs import brightness_service_pb2_grpc
from .service.brightness_servicer import BrightnessServicer

log_format = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)

log = logging.getLogger(__name__)


def serve():
    log.info(f"starting up gRPC server on port {service_port}")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    brightness_service_pb2_grpc.add_BrightnessServiceServicer_to_server(BrightnessServicer(), server)
    server.add_insecure_port(f"[::]:{service_port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
