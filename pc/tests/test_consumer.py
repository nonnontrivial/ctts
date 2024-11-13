from unittest.mock import AsyncMock, patch

import pytest
import asyncpg
from aio_pika import Message

from pc.consumer.consumer import Consumer

@pytest.fixture
async def mock_asyncpg_pool():
    with patch("asyncpg.create_pool") as mock_create_pool:
        mock_pool = AsyncMock()
        mock_create_pool.return_value = mock_pool

        mock_connection = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        yield mock_pool

amqp_url="amqp://localhost"

@pytest.fixture
def consumer(mock_asyncpg_pool):
    prediction_queue="prediction"
    cycle_queue="cycle"
    return Consumer(
        url=amqp_url,
        prediction_queue=prediction_queue,
        cycle_queue=cycle_queue,
        connection_pool=mock_asyncpg_pool,
        on_cycle_completion=lambda _: None
    )

@pytest.mark.asyncio
async def test_consumer_connection(consumer):
    with patch("pc.consumer.consumer.Consumer.connect", new_callable=AsyncMock) as mock_connect:
        mock_connection = AsyncMock()
        mock_channel = AsyncMock()
        mock_connect.return_value = mock_connection
        mock_connection.channel.return_value = mock_channel
        await consumer.connect()
        mock_connect.assert_called_once()
