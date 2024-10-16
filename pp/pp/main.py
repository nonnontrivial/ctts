import logging

import pika
from pika.exceptions import AMQPConnectionError

from .cells.continent_manager import H3ContinentManager
from .config import rabbitmq_host, queue_name, api_port, api_host
from .publisher.cell_prediction_publisher import CellPredictionPublisher

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main():
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(rabbitmq_host))
        pika_channel = connection.channel()
        pika_channel.queue_declare(queue=queue_name)
    except AMQPConnectionError as _:
        import sys

        log.error(f"could not form amqp connection; has rabbitmq started?")
        log.warning("exiting")
        sys.exit(1)
    else:
        continent_manager = H3ContinentManager(continent="north-america")
        cell_publisher = CellPredictionPublisher(continent_manager=continent_manager, api_host=api_host,
                                                 api_port=api_port,
                                                 channel=pika_channel, queue_name=queue_name)
        try:
            cell_publisher.publish()
        except Exception as e:
            log.error(f"unable to publish cell predictions {e}")
            pika_channel.close()


if __name__ == "__main__":
    main()
