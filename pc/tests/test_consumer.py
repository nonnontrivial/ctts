import asyncio
from unittest.mock import AsyncMock, patch

import pytest
import asyncpg

from pc.consumer.consumer import Consumer

from .fixtures import *

@pytest.mark.asyncio
async def test_consumer_can_consume_from_queues(consumer: Consumer, mock_channel, mock_queues):
    mock_channel.declare_queue.side_effect = [
        mock_queues["prediction"],
        mock_queues["cycle"],
    ]
    task = asyncio.create_task(consumer.consume_from_queues())
    await asyncio.sleep(0.1)
    task.cancel()
    mock_channel.declare_queue.assert_any_call("prediction")
    mock_channel.declare_queue.assert_any_call("cycle")
    mock_queues["prediction"].consume.assert_called_once()
    mock_queues["cycle"].consume.assert_called_once()
