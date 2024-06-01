import os

rabbitmq_host = os.getenv("RABBITMQ_HOST", "localhost")
prediction_queue = os.getenv("PREDICTION_QUEUE", "prediction")

sleep_interval = float(os.getenv("SLEEP_INTERVAL", "0.5"))

api_protocol = os.getenv("API_PROTOCOL", "http")
api_port = int(os.getenv("API_PORT", "8000"))
api_host = os.getenv("API_HOST", "localhost")
api_version = os.getenv("API_VERSION", "v1")
