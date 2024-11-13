from unittest.mock import MagicMock
from datetime import datetime, timedelta
import uuid

import pytest

from pp.cells.cell_publisher import CellPublisher
from pp.cells.cell_covering import CellCovering


@pytest.fixture
def mock_grpc_client(mocker):
    from datetime import datetime, timezone
    from pp.stubs.brightness_service_pb2 import BrightnessObservation

    mock_client_stub = mocker.MagicMock()

    mock_brightness_observation = BrightnessObservation()
    mock_brightness_observation.uuid = str(uuid.uuid4())
    mock_brightness_observation.utc_iso = datetime.now(timezone.utc).isoformat()
    mock_brightness_observation.mpsas = 10.

    mock_client_stub.GetBrightnessObservation.return_value = mock_brightness_observation

    mocker.patch("pp.cells.cell_publisher.BrightnessServiceStub", return_value=mock_client_stub)
    return mock_client_stub


@pytest.fixture
def mock_pika_channel(mocker):
    channel_mock = MagicMock()
    connection_mock = MagicMock()
    connection_mock.channel.return_value = channel_mock
    mocker.patch("pika.BlockingConnection", return_value=connection_mock)
    return channel_mock


@pytest.fixture
def publisher(mock_grpc_client, mock_pika_channel):
    return CellPublisher(api_host="localhost",
                         api_port=50051,
                         channel=mock_pika_channel,
                         prediction_queue="prediction",
                         cycle_queue="cycle")

def test_brightness_message_publish(publisher, mock_pika_channel):
    cell = "89283082813ffff"
    publisher.publish_cell_brightness_message(cell)
    mock_pika_channel.basic_publish.assert_called_once()

def test_cycle_completion_message_publish(publisher, mock_pika_channel):
    then = datetime.now() - timedelta(minutes=5)
    now = datetime.now()
    publisher.publish_cycle_completion_message(then, now)
    mock_pika_channel.basic_publish.assert_called_once()
