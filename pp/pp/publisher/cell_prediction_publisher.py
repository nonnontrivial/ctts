import logging
import json
import typing
from datetime import datetime
from collections import defaultdict

import grpc
from h3 import h3_to_geo
from pika.adapters.blocking_connection import BlockingChannel

from ..cells.cell_covering import H3CellCovering, get_cell_id
from ..stubs.brightness_service_pb2_grpc import BrightnessServiceStub
from ..stubs import brightness_service_pb2
from ..models.models import BrightnessObservation, CellCycle

log = logging.getLogger(__name__)


class CellPredictionPublisher:
    cell_counts = defaultdict(int)

    def __init__(self, cell_covering: H3CellCovering, api_host: str, api_port: int, channel: BlockingChannel,
                 prediction_queue: str, cycle_queue: str):
        self._cell_covering = cell_covering
        self._prediction_queue = prediction_queue
        self._cycle_queue = cycle_queue
        self._channel = channel

        grpc_channel = grpc.insecure_channel(f"{api_host}:{api_port}")
        stub = BrightnessServiceStub(grpc_channel)
        self._stub = stub

    def _publish(self, queue_name: str, message: typing.Dict[str, typing.Any]):
        """publish a message onto a queue"""
        self._channel.basic_publish(exchange="", routing_key=queue_name, body=json.dumps(message))

    def predict_cell_brightness(self, cell) -> None:
        """ask brightness service for prediction of sky brightness on h3 cell
        for the current time"""
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
                h3_id=get_cell_id(lat, lon, resolution=6),
                utc_iso=response.utc_iso,
                mpsas=response.mpsas,
            )
            self._publish(self._prediction_queue, brightness_observation.model_dump())

    def run(self):
        while True:
            start = datetime.now()
            for cell in self._cell_covering():
                CellPredictionPublisher.cell_counts[cell] += 1

                self.predict_cell_brightness(cell)
                log.debug(f"{len(CellPredictionPublisher.cell_counts)} distinct cells have had observations published")

            end = datetime.now()
            cell_cycle = CellCycle(start=start, end=end, duration_s=int((end - start).total_seconds()))
            self._publish(self._cycle_queue, cell_cycle.model_dump())
