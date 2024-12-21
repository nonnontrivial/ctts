from datetime import datetime, timedelta

import pytest
from .fixtures import *

@pytest.mark.parametrize("cell_id", [
    ("89283082813ffff"),
    ("8928308280fffff"),
    ("89283082807ffff"),
])
def test_can_publish_cell_brightness(cell_id, publisher, mock_pika_channel):
    publisher.publish_cell_brightness_message(cell_id)
    mock_pika_channel.basic_publish.assert_called_once()

@pytest.mark.parametrize("minutes_ago", [
    (i) for i in range(1, 10)
])
def test_can_publish_cycle_complete(minutes_ago, publisher, mock_pika_channel):
    then = datetime.now() - timedelta(minutes=minutes_ago)
    now = datetime.now()
    publisher.publish_cycle_completion_message(then, now)
    mock_pika_channel.basic_publish.assert_called_once()
