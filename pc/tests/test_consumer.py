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

@pytest.fixture
def consumer(mock_asyncpg_pool):
    amqp_url="amqp://localhost"
    prediction_queue="prediction"
    return Consumer(url=amqp_url, prediction_queue=prediction_queue,cycle_queue="",connection_pool=mock_asyncpg_pool)

@pytest.mark.asyncio
async def test_consumer(consumer):
    assert consumer is not None
