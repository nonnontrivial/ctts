from datetime import datetime, timedelta

import h3
import pytest
from .fixtures import *

@pytest.mark.parametrize("cell_id", [x for x in h3.get_res0_indexes()])
def test_can_publish_cell_brightness_at_h3_indexes(cell_id, cell_publisher, mock_pika_channel):
    cell_publisher.publish_cell_brightness_message(cell_id)
    mock_pika_channel.basic_publish.assert_called_once()

@pytest.mark.parametrize("minutes_ago", [i for i in range(1, 10)])
def test_can_publish_cycle_complete(minutes_ago, cell_publisher, mock_pika_channel):
    then = datetime.now() - timedelta(minutes=minutes_ago)
    now = datetime.now()
    cell_publisher.publish_cycle_completion_message(then, now)
    mock_pika_channel.basic_publish.assert_called_once()

def test_publisher_raises_on_empty_cells(mock_grpc_client, mock_pika_channel):
    with pytest.raises(ValueError):
        cell_publisher = CellPublisher(
            api_host="localhost",
            api_port=50051,
            channel=mock_pika_channel,
            prediction_queue="prediction",
            cycle_queue="cycle",
            path_to_geojson=Path(__file__).parent / "empty.geojson"
        )
        cell_publisher.run()
