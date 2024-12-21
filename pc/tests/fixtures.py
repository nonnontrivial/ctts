from unittest.mock import AsyncMock, patch

import pytest
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
    prediction_queue="prediction"
    cycle_queue="cycle"

    return Consumer(
        url="amqp://localhost",
        prediction_queue=prediction_queue,
        cycle_queue=cycle_queue,
        connection_pool=mock_asyncpg_pool,
        on_cycle_completion=lambda _: None
    )
