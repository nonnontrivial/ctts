import logging
import asyncio

import httpx
import pika

from .prediction import predict_on_cell
from .cells import get_res_zero_cell_coords
from .config import rabbitmq_host, prediction_queue, sleep_interval

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


async def main():
    """initializes process of getting sky brightness predictions for h3 cells;
    publishing them to prediction queue as available.

    over 122 res 0 cells on current machine, this setup will publish at a rate of 1.4m/s
    """

    connection = pika.BlockingConnection(pika.ConnectionParameters(rabbitmq_host))
    channel = connection.channel()
    channel.queue_declare(queue=prediction_queue)

    all_h3_cell_coords = get_res_zero_cell_coords()
    log.info(f"using {len(all_h3_cell_coords)} resolution zero cells")

    async with httpx.AsyncClient() as client:
        while True:
            for coords in all_h3_cell_coords:
                await asyncio.create_task(predict_on_cell(client, coords, channel))
                await asyncio.sleep(sleep_interval)
            # tasks = [predict_on_cell(client, coords, channel) for coords in all_h3_cell_coords]
            # await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
