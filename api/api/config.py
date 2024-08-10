import os

api_version = os.getenv("API_VERSION", "v1")

service_port = int(os.getenv("SERVICE_PORT", 8000))
service_host = "0.0.0.0"

log_level = int(os.getenv("LOG_LEVEL", 20))
