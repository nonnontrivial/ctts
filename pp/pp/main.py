import logging

import pika
from pika.adapters.blocking_connection import BlockingChannel
from pika.exceptions import AMQPConnectionError

from .config import rabbitmq_host, prediction_queue, cycle_queue, api_port, api_host
from .cells.cell_publisher import CellPublisher

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def run_publisher():
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(rabbitmq_host))

        pika_channel = connection.channel()
        pika_channel.queue_declare(queue=prediction_queue)
        pika_channel.queue_declare(queue=cycle_queue)

        cell_publisher = CellPublisher(api_host=api_host,
                                       api_port=api_port,
                                       channel=pika_channel,
                                       prediction_queue=prediction_queue,
                                       cycle_queue=cycle_queue)
    except AMQPConnectionError as _:
        import sys

        log.error(f"could not form amqp connection; is rabbitmq running?")
        log.warning("exiting")
        sys.exit(1)
    except Exception as e:
        log.error(f"could not start amqp connection: {e}")
    else:
        try:
            log.info("running publisher")
            cell_publisher.run()
        except Exception as e:
            log.error(f"unable to publish cell predictions: {e}")
            pika_channel.close()


if __name__ == "__main__":
    run_publisher()
