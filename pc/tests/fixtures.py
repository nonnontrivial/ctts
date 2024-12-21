from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pc.consumer.consumer import Consumer


@pytest.fixture
def mock_connection():
    connection = AsyncMock()
    channel = AsyncMock()
    connection.channel.return_value = channel
    return connection

@pytest.fixture
def mock_channel(mock_connection):
    return mock_connection.channel.return_value

@pytest.fixture
def mock_queues():
    prediction_queue = AsyncMock()
    cycle_queue = AsyncMock()
    return {"prediction": prediction_queue, "cycle": cycle_queue}

@pytest.fixture
def mock_pool():
    return AsyncMock()

@pytest.fixture
def mock_handler():
    return AsyncMock()

@pytest.fixture
def mock_shutdown():
    return MagicMock()

@pytest.fixture
def consumer(mock_connection, mock_pool, mock_handler, mock_shutdown):
    consumer = Consumer(
        url="amqp://test",
        prediction_queue="prediction",
        cycle_queue="cycle",
        connection_pool=mock_pool,
        on_cycle_completion=mock_handler,
    )
    consumer.connection = mock_connection
    return consumer
