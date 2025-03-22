import os

resolution = int(os.environ.get("RESOLUTION", 0))
api_host = os.environ.get("API_HOST", "localhost")
api_port = int(os.environ.get("API_PORT", 8000))

rabbitmq_user = os.getenv("RABBITMQ_USER", "guest")
rabbitmq_password = os.getenv("RABBITMQ_PASSWORD", "guest")
rabbitmq_host = os.getenv("RABBITMQ_HOST", "localhost")
queue = os.getenv("QUEUE", "brightness.snapshot")

client_timeout_seconds = int(os.getenv("CLIENT_TIMEOUT_SECONDS", 60))

broker_url = f"amqp://{rabbitmq_user}:{rabbitmq_password}@{rabbitmq_host}"
geojson_url = f"http://{api_host}:{api_port}/geojson"
inference_url = f"http://{api_host}:{api_port}/infer"
