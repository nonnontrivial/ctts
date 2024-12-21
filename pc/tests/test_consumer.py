from unittest.mock import AsyncMock, patch

import pytest
import asyncpg

from .fixtures import *

@patch("pc.consumer.consumer.Consumer.connect", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_consumer_can_connect(mock_connect, consumer):
    mock_connection = AsyncMock()
    mock_channel = AsyncMock()
    mock_connect.return_value = mock_connection
    mock_connection.channel.return_value = mock_channel
    await consumer.connect()
    mock_connect.assert_called_once()
