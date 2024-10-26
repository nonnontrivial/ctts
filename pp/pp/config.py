import os

rabbitmq_host = os.getenv("RABBITMQ_HOST", "localhost")
prediction_queue = os.getenv("PREDICTION_QUEUE", "brightness.prediction")
cycle_queue = os.getenv("CYCLE_QUEUE", "brightness.cycle")

api_protocol = os.getenv("API_PROTOCOL", "http")
api_host = os.getenv("API_HOST", "localhost")
api_port = int(os.getenv("API_PORT", 50051))
