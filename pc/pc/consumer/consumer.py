import json
import logging
import asyncio
import typing

import asyncpg
import aio_pika
from aio_pika.abc import AbstractIncomingMessage, AbstractRobustChannel

from pc.persistence.models import BrightnessObservation
from pc.persistence.db import insert_brightness_observation

log = logging.getLogger(__name__)


class Consumer:
    def __init__(self, url: str, prediction_queue: str, cycle_queue: str, connection_pool: asyncpg.Pool):
        self._amqp_url = url
        self._prediction_queue = prediction_queue
        self._cycle_queue = cycle_queue
        self._pool = connection_pool

    async def start(self):
        try:
            log.info(f"connecting to {self._amqp_url}")
            connection = await aio_pika.connect_robust(self._amqp_url)
        except Exception as e:
            import sys

            log.error(f"could not form amqp connection because {e}; is rabbitmq running?")
            log.warning("exiting")
            sys.exit(1)
        else:
            async with connection:
                channel = await connection.channel()
                await self.consume(channel) # type: ignore

    async def consume(self, channel: AbstractRobustChannel):
        log.info(f"consuming from {self._prediction_queue}")
        prediction_queue = await channel.declare_queue(self._prediction_queue)
        await prediction_queue.consume(self._on_prediction_message, no_ack=True)

        # cycle_queue = await channel.declare_queue(self._cycle_queue)
        # await cycle_queue.consume(self._on_cycle_message, no_ack=True)

        await asyncio.Future()

    async def _on_cycle_message(self, message: AbstractIncomingMessage):
        """handle incoming message by retrieving max reading from postgres within
        the range in the mesage"""
        log.debug(f"received message {message.body}")

    async def _on_prediction_message(self, message: AbstractIncomingMessage):
        """handle incoming message by storing in postgres"""
        try:
            log.debug(f"received message {message.body}")
            message_dict: typing.Dict = json.loads(message.body.decode())
            brightness = BrightnessObservation(**message_dict)
            await insert_brightness_observation(self._pool, brightness)
        except Exception as e:
            log.error(f"could not save brightness observation: {e}")
        else:
            log.info(f"saved brightness of {brightness.h3_id}")
