import json
import logging
import asyncio
import typing

import aio_pika
from aio_pika.abc import AbstractIncomingMessage, AbstractRobustChannel

from pc.persistence.models import BrightnessObservation
from pc.consumer.websocket_handler import WebsocketHandler

log = logging.getLogger(__name__)


class Consumer:
    def __init__(self, url: str, queue_name: str, websocket_handler: typing.Optional[WebsocketHandler] = None):
        self._amqp_url = url
        self._queue_name = queue_name
        self._websocket_handler = websocket_handler

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
        """begin consuming from queue on a given channel"""
        queue = await channel.declare_queue(self._queue_name)
        await queue.consume(self._on_message, no_ack=True)
        await asyncio.Future()

    async def _on_message(self, message: AbstractIncomingMessage):
        """handle incoming message by storing in postgres"""
        try:
            log.info(f"received message {message.body}")

            brightness_observation_json = json.loads(message.body.decode())
            brightness_observation = BrightnessObservation(**brightness_observation_json)

            await brightness_observation.save()

            if self._websocket_handler is not None:
                await self._websocket_handler.broadcast(brightness_observation_json)
        except Exception as e:
            log.error(f"could not save brightness observation {e}")
        else:
            log.info(f"saved {brightness_observation}")
