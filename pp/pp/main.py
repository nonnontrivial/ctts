import logging
import asyncio

import httpx
import pika
from pika.exceptions import AMQPConnectionError

from .prediction import predict_on_cell_coords
from .cells import get_res_zero_cell_coords
from .config import rabbitmq_host, prediction_queue, task_sleep_interval

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


async def main():
    """initializes process of getting sky brightness predictions for h3 cells;
    publishing them to prediction queue as available.

    n.b. with 122 res 0 cells on 2016 macbook, this will publish at a rate of 1.4m/s
    """

    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(rabbitmq_host))
        channel = connection.channel()
        channel.queue_declare(queue=prediction_queue)
    except AMQPConnectionError as e:
        import sys

        log.error(f"could not form amqp connection {e}")
        log.warning("exiting")
        sys.exit(1)
    except Exception as e:
        log.error(f"could not start publisher because {e}")
    else:
        resolution_zero_cell_coords = get_res_zero_cell_coords()
        log.debug(f"using {len(resolution_zero_cell_coords)} resolution zero cells")

        try:
            async with httpx.AsyncClient() as client:
                while True:
                    for cell_coords in resolution_zero_cell_coords:
                        await asyncio.create_task(predict_on_cell_coords(client, cell_coords, channel))
                        await asyncio.sleep(task_sleep_interval)
        except Exception as e:
            log.error(f"could not continue publishing because {e}")
            channel.close()


if __name__ == "__main__":
    asyncio.run(main())
