import json
import logging
import asyncio
import typing

import aio_pika
from aio_pika.abc import AbstractIncomingMessage, AbstractRobustChannel

from pc.persistence.models import BrightnessObservation

log = logging.getLogger(__name__)


class Consumer:
    def __init__(self, url: str, prediction_queue: str, cycle_queue: str):
        self._amqp_url = url
        self._prediction_queue = prediction_queue
        self._cycle_queue = cycle_queue

    async def start(self):
        try:
            connection = await aio_pika.connect_robust(self._amqp_url)
        except Exception as e:
            import sys

            log.error(f"could not form amqp connection because {e}; has rabbitmq started?")
            log.warning("exiting")
            sys.exit(1)
        else:
            async with connection:
                channel = await connection.channel()
                await self.consume(channel) # type: ignore

    async def consume(self, channel: AbstractRobustChannel):
        """consume from the queues we care about"""
        prediction_queue = await channel.declare_queue(self._prediction_queue)
        cycle_queue = await channel.declare_queue(self._cycle_queue)

        await prediction_queue.consume(self._on_prediction_message, no_ack=True)
        await cycle_queue.consume(self._on_cycle_message, no_ack=True)

        await asyncio.Future()

    async def _on_cycle_message(self, message: AbstractIncomingMessage):
        """handle incoming message by retrieving max sqm reading from postgres within the range"""
        pass

    async def _on_prediction_message(self, message: AbstractIncomingMessage):
        """handle incoming message by storing in postgres"""
        try:
            log.debug(f"received message {message.body}")
            brightness_observation_json = json.loads(message.body.decode())
            brightness_observation = BrightnessObservation(**brightness_observation_json)

            await brightness_observation.save()
        except Exception as e:
            log.error(f"could not save brightness observation {e}")
        else:
            log.info(f"saved {brightness_observation}")
