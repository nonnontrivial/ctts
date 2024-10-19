from unittest import mock

import pytest
from aio_pika import Message

from pc.consumer.consumer import Consumer

@pytest.fixture
def consumer():
    amqp_url="amqp://localhost"
    prediction_queue="prediction"
    return Consumer(url=amqp_url, queue_name=prediction_queue)

@pytest.mark.skip
@pytest.mark.asyncio
async def test_can_consume_message(consumer):
    pass
