import json
import logging
import asyncio

import aio_pika

from pc.config import *
from pc.persistence.models import BrightnessObservation

log = logging.getLogger(__name__)


# class RabbitMQConsumer:
#     def __init__(self, user: str, password: str, host: str):
#         self.url = f"amqp://{user}:{password}@{host}"
#
#     async def connect(self):
#         pass


async def consume_brightness_observations():
    """begin consuming messages from the queue, storing them in predictions table"""
    try:
        amqp_connection = await aio_pika.connect_robust(f"amqp://{AMQP_USER}:{AMQP_PASSWORD}@{AMQP_HOST}")
    except Exception as e:
        import sys

        log.error(f"could not form amqp connection because {e}; has rabbitmq started?")
        log.warning("exiting")
        sys.exit(1)
    else:
        async with amqp_connection:
            channel = await amqp_connection.channel()
            queue = await channel.declare_queue(AMQP_PREDICTION_QUEUE)

            async for message in queue:
                async with message.process():
                    brightness_observation_json = json.loads(message.body.decode())
                    brightness_observation = BrightnessObservation(**brightness_observation_json)

                    log.info(f"saving brightness observation {brightness_observation}")
                    await brightness_observation.save()
        await asyncio.Future()
