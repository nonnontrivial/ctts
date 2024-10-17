from unittest.mock import MagicMock
import uuid

import pytest

from pp.publisher.cell_prediction_publisher import CellPredictionPublisher
from pp.cells.continent_manager import H3ContinentManager


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

    mocker.patch("pp.publisher.cell_prediction_publisher.BrightnessServiceStub", return_value=mock_client_stub)
    return mock_client_stub


@pytest.fixture
def mock_pika_channel(mocker):
    channel_mock = MagicMock()
    connection_mock = MagicMock()
    connection_mock.channel.return_value = channel_mock
    mocker.patch("pika.BlockingConnection", return_value=connection_mock)
    return channel_mock


def test_publisher_publishes_on_channel(mock_grpc_client, mock_pika_channel):
    continent_manager = H3ContinentManager(continent="north-america")
    cell_publisher = CellPredictionPublisher(continent_manager=continent_manager, api_host="localhost",
                                             api_port=50051,
                                             channel=mock_pika_channel, queue_name="prediction")

    cell_publisher.publish_prediction_at_cell("89283082813ffff")
    mock_pika_channel.basic_publish.assert_called_once()
