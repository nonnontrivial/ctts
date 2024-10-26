import os

pg_user = os.getenv("PG_USER", "postgres")
pg_password = os.getenv("PG_PASSWORD", "password")
pg_database = os.getenv("PG_DATABASE", "localhost")
pg_host = os.getenv("PG_HOST", "postgres")
pg_port = int(os.getenv("PG_PORT", 5432))

pg_dsn = f"postgres://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_database}"

rabbitmq_user = os.getenv("AMQP_USER", "guest")
rabbitmq_password = os.getenv("AMQP_PASSWORD", "guest")
rabbitmq_host = os.getenv("AMQP_HOST", "localhost")
prediction_queue = os.getenv("AMQP_PREDICTION_QUEUE", " brightness.prediction")
cycle_queue = os.getenv("AMQP_CYCLE_QUEUE", " brightness.cycle")

amqp_url = f"amqp://{rabbitmq_user}:{rabbitmq_password}@{rabbitmq_host}"

ws_host = os.getenv("WS_HOST", "0.0.0.0")
ws_port = int(os.getenv("WS_PORT", 8765))
