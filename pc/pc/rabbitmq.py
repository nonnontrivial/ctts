import json
import logging
import asyncio

import aio_pika

from pc.config import *
from pc.persistence.models import BrightnessObservation

log = logging.getLogger(__name__)


async def consume_brightness_observations():
    """begin consuming messages from the queue"""
    try:
        protocol = "amqp"
        amqp_connection = await aio_pika.connect_robust(f"{protocol}://{AMQP_USER}:{AMQP_PASSWORD}@{AMQP_HOST}")
    except Exception as e:
        import sys

        log.error(f"could not form amqp connection because {e}; has rabbitmq started?")
        log.warning("exiting")
        sys.exit(1)
    else:
        async with amqp_connection:
            channel = await amqp_connection.channel()
            queue = await channel.declare_queue(AMQP_PREDICTION_QUEUE)
            await queue.consume(save_brightness_observation)
            await asyncio.Future()


async def save_brightness_observation(message: aio_pika.IncomingMessage):
    """store brightness message in `brightnessobservation` table"""
    async with message.process():
        brightness_observation_json = json.loads(message.body.decode())
        brightness_observation = BrightnessObservation(**brightness_observation_json)

        try:
            await brightness_observation.save()
        except Exception as e:
            log.error(f"could not save brightness observation {e}")
        else:
            log.info(f"saved brightness observation {brightness_observation}")
