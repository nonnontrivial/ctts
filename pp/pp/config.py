import os

keydb_host = os.getenv("KEYDB_HOST", "keydb")
keydb_port = int(os.getenv("KEYDB_PORT", 6379))
rabbitmq_host = os.getenv("RABBITMQ_HOST", "localhost")
prediction_queue = os.getenv("PREDICTION_QUEUE", "prediction")

task_sleep_interval = float(os.getenv("SLEEP_INTERVAL", "0.5"))

api_protocol = os.getenv("API_PROTOCOL", "http")
api_port = int(os.getenv("API_PORT", "8000"))
api_host = os.getenv("API_HOST", "localhost")
api_version = os.getenv("API_VERSION", "v1")
model_version = os.getenv("MODEL_VERSION", "0.1.0")
