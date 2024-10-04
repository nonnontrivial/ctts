import os

PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "password")
PG_DATABASE = os.getenv("PG_DATABASE", "localhost")
PG_HOST = os.getenv("PG_HOST", "postgres")
PG_PORT = int(os.getenv("PG_PORT", 5432))
pg_dsn = f"postgres://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"

rabbitmq_user = os.getenv("AMQP_USER", "guest")
rabbitmq_password = os.getenv("AMQP_PASSWORD", "guest")
rabbitmq_host = os.getenv("AMQP_HOST", "localhost")
prediction_queue_name = os.getenv("AMQP_PREDICTION_QUEUE", "prediction")
