import json
import logging
import asyncio
import typing

import asyncpg
import aio_pika
from aio_pika.abc import AbstractIncomingMessage, AbstractRobustChannel

from pc.persistence.models import BrightnessObservation, CellCycle
from pc.persistence.db import insert_brightness_observation, select_max_brightness_record_in_range

log = logging.getLogger(__name__)


class Consumer:
    def __init__(self, url: str, prediction_queue: str, cycle_queue: str, connection_pool: asyncpg.Pool, on_cycle_completion: typing.Callable[[BrightnessObservation],None]):
        self._amqp_url = url
        self._prediction_queue = prediction_queue
        self._cycle_queue = cycle_queue
        self._pool = connection_pool
        self._on_cycle_completion = on_cycle_completion

        self.connection = None

    async def connect(self):
        """establish the connection or exit"""
        try:
            log.info(f"connecting to {self._amqp_url}")
            self.connection = await aio_pika.connect_robust(self._amqp_url)
        except Exception as e:
            import sys

            log.error(f"could not form amqp connection because {e}; is rabbitmq running?")
            log.warning("exiting")
            sys.exit(1)

    async def consume_from_queues(self):
        """consume data from the prediction and cycle queues"""
        if self.connection is None:
            raise ValueError("there is no connection!")

        async with self.connection:
            channel = await self.connection.channel()
            queues = {
                self._prediction_queue: self._on_prediction_message,
                self._cycle_queue: self._on_cycle_message
            }

            for queue_name, handler in queues.items():
                log.info(f"consuming from {queue_name}")
                queue = await channel.declare_queue(queue_name)
                await queue.consume(handler, no_ack=True)

            log.info("waiting on messages")
            await asyncio.Future()

    async def _on_cycle_message(self, message: AbstractIncomingMessage):
        """handle incoming message by retrieving max reading from postgres within
        the range in the mesage"""
        log.debug(f"received message {message.body}")
        try:
            message_dict: typing.Dict = json.loads(message.body.decode())
            cell_cycle = CellCycle(**message_dict)

            record = await select_max_brightness_record_in_range(self._pool, cell_cycle)
            if record is None:
                raise ValueError(f"no record in range of cycle {cell_cycle}")

            record = dict(record)
            uuid = str(record["uuid"])
            del record["uuid"]
            max_brightness_observation_in_cycle = BrightnessObservation(**record, uuid=uuid)
            self._on_cycle_completion(max_brightness_observation_in_cycle)
        except Exception as e:
            log.error(f"could not process cycle message: {e}")


    async def _on_prediction_message(self, message: AbstractIncomingMessage):
        """handle incoming message by storing in postgres"""
        log.debug(f"received message {message.body}")
        try:
            message_dict: typing.Dict = json.loads(message.body.decode())
            brightness_observation = BrightnessObservation(**message_dict)
            await insert_brightness_observation(self._pool, brightness_observation)
        except Exception as e:
            log.error(f"could not save brightness observation: {e}")
        else:
            log.debug(f"saved brightness of {brightness_observation.h3_id}")
