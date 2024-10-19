import logging
import json
from collections import defaultdict

import grpc
from h3 import h3_to_geo
from pika.adapters.blocking_connection import BlockingChannel

from ..cells.continent_manager import H3ContinentManager
from ..stubs.brightness_service_pb2_grpc import BrightnessServiceStub
from ..stubs import brightness_service_pb2
from ..models.models import BrightnessObservation

log = logging.getLogger(__name__)


class CellPredictionPublisher:
    cell_counts = defaultdict(int)

    def __init__(self, continent_manager: H3ContinentManager, api_host: str, api_port: int, channel: BlockingChannel,
                 queue_name: str):
        self._continent_manager = continent_manager
        self._queue_name = queue_name
        self._channel = channel

        grpc_channel = grpc.insecure_channel(f"{api_host}:{api_port}")
        stub = BrightnessServiceStub(grpc_channel)
        self._stub = stub

    def publish_prediction_at_cell(self, cell):
        lat, lon = h3_to_geo(cell)
        request = brightness_service_pb2.Coordinates(lat=lat, lon=lon)
        try:
            response = self._stub.GetBrightnessObservation(request)
        except grpc.RpcError as e:
            log.error(f"rpc error on brightness requests {e}")
        else:
            log.info(f"brightness observation response for {cell} is {response}")
            brightness_observation = BrightnessObservation(
                uuid=response.uuid,
                lat=lat,
                lon=lon,
                h3_id=self._continent_manager.get_cell_id(lat, lon),
                utc_iso=response.utc_iso,
                mpsas=response.mpsas,
            )
            self._channel.basic_publish(exchange="", routing_key=self._queue_name,
                                        body=json.dumps(brightness_observation.model_dump()))

    def publish(self):
        """get brightness observations for a set of cells, forwarding responses to rabbitmq"""
        cells = self._continent_manager.get_cell_covering()
        while True:
            for cell in cells:
                CellPredictionPublisher.cell_counts[cell] += 1
                self.publish_prediction_at_cell(cell)
                log.debug(f"{len(CellPredictionPublisher.cell_counts)} distinct cells have had observations published")
