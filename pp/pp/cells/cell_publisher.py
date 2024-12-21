import logging
import json
import typing
from datetime import datetime, timezone
from collections import defaultdict

import grpc
from h3 import h3_to_geo
from pika.adapters.blocking_connection import BlockingChannel

from ..cells.cell_covering import CellCovering
from ..config import resolution
from ..stubs.brightness_service_pb2_grpc import BrightnessServiceStub
from ..stubs import brightness_service_pb2
from ..models.models import BrightnessObservation, CellCycle

log = logging.getLogger(__name__)


class CellPublisher(CellCovering):
    cell_counts = defaultdict(int)

    def __init__(self, api_host: str, api_port: int, channel: BlockingChannel,
                 prediction_queue: str, cycle_queue: str):
        super().__init__()

        self._prediction_queue = prediction_queue
        self._cycle_queue = cycle_queue
        self._channel = channel

        grpc_channel = grpc.insecure_channel(f"{api_host}:{api_port}")
        stub = BrightnessServiceStub(grpc_channel)
        self._stub = stub

    def _publish(self, queue_name: str, message: typing.Dict[str, typing.Any]):
        """publish a message onto a queue"""
        log.info(f"publishing {message} to {queue_name}")
        self._channel.basic_publish(exchange="", routing_key=queue_name, body=json.dumps(message))

    def publish_cell_brightness_message(self, cell) -> None:
        lat, lon = h3_to_geo(cell)
        request = brightness_service_pb2.Coordinates(lat=lat, lon=lon)
        try:
            response = self._stub.GetBrightnessObservation(request)
        except grpc.RpcError as e:
            log.error(f"rpc error on brightness requests {e}")
        else:
            log.debug(f"brightness observation response for {cell} is {response}")
            brightness_observation = BrightnessObservation(
                uuid=response.uuid,
                lat=lat,
                lon=lon,
                h3_id=CellCovering.get_cell_id(lat, lon, resolution=resolution),
                mpsas=response.mpsas,
                timestamp_utc=response.utc_iso,
            )
            dumped = brightness_observation.model_dump()
            dumped["timestamp_utc"] = brightness_observation.timestamp_utc.isoformat()
            self._publish(self._prediction_queue, dumped)

    def publish_cycle_completion_message(self, start: datetime, end: datetime) -> None:
        cell_cycle = CellCycle(start_time_utc=start, end_time_utc=end, duration_s=int((end - start).total_seconds()))
        cell_cycle = cell_cycle.model_dump()
        cell_cycle["start_time_utc"] = cell_cycle["start_time_utc"].isoformat()
        cell_cycle["end_time_utc"] = cell_cycle["end_time_utc"].isoformat()
        self._publish(self._cycle_queue, cell_cycle)

    def run(self):
        cells = self.covering
        if len(cells) == 0:
            raise ValueError("cell covering is empty!")
        log.info(f"publishing brightness for {len(cells)} cells(s)")

        while True:
            start_time_utc = datetime.now(timezone.utc)

            for cell in cells:
                CellPublisher.cell_counts[cell] += 1
                self.publish_cell_brightness_message(cell)
                log.debug(f"{len(CellPublisher.cell_counts)} distinct cells have had observations published")

            end_time_utc = datetime.now(timezone.utc)
            self.publish_cycle_completion_message(start_time_utc, end_time_utc)
