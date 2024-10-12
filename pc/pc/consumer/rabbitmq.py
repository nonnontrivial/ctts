import json
import logging
import asyncio

import aio_pika
from aio_pika.abc import AbstractIncomingMessage

from pc.config import amqp_url, prediction_queue
from pc.persistence.models import BrightnessObservation
from pc.consumer.websocket_handler import ws_handler

log = logging.getLogger(__name__)


async def consume_brightness_observations():
    """consume messages from the prediction queue"""
    try:
        connection = await aio_pika.connect_robust(amqp_url)
    except Exception as e:
        import sys

        log.error(f"could not form amqp connection because {e}; has rabbitmq started?")
        log.warning("exiting")
        sys.exit(1)
    else:
        async with connection:
            channel = await connection.channel()
            queue = await channel.declare_queue(prediction_queue)
            await queue.consume(ingest_brightness_message, no_ack=True)
            await asyncio.Future()


async def ingest_brightness_message(message: AbstractIncomingMessage):
    """store and disseminate brightness message"""
    log.info(f"received message {message.body}")

    try:
        brightness_observation_json = json.loads(message.body.decode())
        brightness_observation = BrightnessObservation(**brightness_observation_json)

        await brightness_observation.save()
        await ws_handler.broadcast(brightness_observation_json)
    except Exception as e:
        log.error(f"could not save brightness observation {e}")
    else:
        log.info(f"saved {brightness_observation}")
