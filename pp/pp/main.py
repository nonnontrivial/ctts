import logging
import typing

import grpc
import pika
from pika.adapters.blocking_connection import BlockingChannel
from pika.exceptions import AMQPConnectionError

from .stubs.brightness_service_pb2_grpc import BrightnessServiceStub
from .cells.continent_manager import H3ContinentManager
from .config import rabbitmq_host, queue_name, api_port, api_host

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

continent_manager = H3ContinentManager(continent="north-america")


def run_observation_requests(h3_cells: typing.Set, pika_channel: BlockingChannel):
    """repeatedly get brightness observations for a set of cells, forwarding
    successful responses onto rabbitmq"""
    import json
    from collections import defaultdict

    from h3 import h3_to_geo

    from .cells.continent_manager import H3ContinentManager
    from .stubs import brightness_service_pb2
    from .models.models import BrightnessObservation

    channel = grpc.insecure_channel(f"{api_host}:{api_port}")
    stub = BrightnessServiceStub(channel)

    cell_counts = defaultdict(int)

    while True:
        for cell in h3_cells:
            cell_counts[cell] += 1

            lat, lon = h3_to_geo(cell)
            request = brightness_service_pb2.BrightnessRequest(lat=lat, lon=lon)

            try:
                response = stub.GetBrightnessObservation(request)
                log.debug(f"brightness observation response is {response}")
            except grpc.RpcError as e:
                log.error(f"rpc error on brightness requests {e}")
            else:
                brightness_observation = BrightnessObservation(
                    uuid=response.uuid,
                    lat=lat,
                    lon=lon,
                    h3_id=H3ContinentManager.get_cell_id(lat, lon),
                    utc_iso=response.utc_iso,
                    mpsas=response.mpsas,
                )
                pika_channel.basic_publish(exchange="", routing_key=queue_name,
                                           body=json.dumps(brightness_observation.model_dump()))
                log.info(f"{len(cell_counts)} distinct cells have had observations published")


def main():
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(rabbitmq_host))
        channel = connection.channel()
        channel.queue_declare(queue=queue_name)
    except AMQPConnectionError as _:
        import sys

        log.error(f"could not form amqp connection; has rabbitmq started?")
        log.warning("exiting")
        sys.exit(1)
    else:
        try:
            na_cells = continent_manager.get_cell_covering()
            log.info(f"requesting predictions for {len(na_cells)} cells")

            run_observation_requests(na_cells, channel)
        except Exception as e:
            log.error(f"unable to run observation requests {e}")
            channel.close()


if __name__ == "__main__":
    main()
