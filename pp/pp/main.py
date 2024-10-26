import logging

import pika
from pika.adapters.blocking_connection import BlockingChannel
from pika.exceptions import AMQPConnectionError

from .config import rabbitmq_host, prediction_queue, cycle_queue, api_port, api_host
from .cells.cell_covering import H3CellCovering
from .publisher.cell_prediction_publisher import CellPredictionPublisher

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def initialize_queues(channel: BlockingChannel):
    channel.queue_declare(queue=prediction_queue)
    channel.queue_declare(queue=cycle_queue)

def main():
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(rabbitmq_host))

        pika_channel = connection.channel()
        initialize_queues(pika_channel)

        cell_covering = H3CellCovering()
        cell_publisher = CellPredictionPublisher(cell_covering=cell_covering, api_host=api_host,
                                                 api_port=api_port,
                                                 channel=pika_channel,
                                                 prediction_queue=prediction_queue,
                                                 cycle_queue=cycle_queue)
    except AMQPConnectionError as _:
        import sys

        log.error(f"could not form amqp connection; has rabbitmq started?")
        log.warning("exiting")
        sys.exit(1)
    except Exception as e:
        log.error(f"could not start amqp connection: {e}")
    else:
        try:
            cell_publisher.run()
        except Exception as e:
            log.error(f"unable to publish cell predictions: {e}")
            pika_channel.close()


if __name__ == "__main__":
    main()
