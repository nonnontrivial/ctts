import json
import logging
import asyncio

import aio_pika
from aio_pika.abc import AbstractIncomingMessage

from pc.config import *
from pc.persistence.models import BrightnessObservation

log = logging.getLogger(__name__)


async def consume_brightness_observations():
    """consume messages from the prediction queue"""
    try:
        connection = await aio_pika.connect_robust(f"amqp://{rabbitmq_user}:{rabbitmq_password}@{rabbitmq_host}")
    except Exception as e:
        import sys

        log.error(f"could not form amqp connection because {e}; has rabbitmq started?")
        log.warning("exiting")
        sys.exit(1)
    else:
        async with connection:
            channel = await connection.channel()
            queue = await channel.declare_queue(prediction_queue_name)
            await queue.consume(save_brightness_observation, no_ack=True)
            await asyncio.Future()


async def save_brightness_observation(message: AbstractIncomingMessage):
    """store brightness message in `brightnessobservation` table"""
    log.info(f"received message {message.body}")

    brightness_observation_json = json.loads(message.body.decode())
    brightness_observation = BrightnessObservation(**brightness_observation_json)

    try:
        await brightness_observation.save()
    except Exception as e:
        log.error(f"could not save brightness observation {e}")
    else:
        log.info(f"saved {brightness_observation}")
