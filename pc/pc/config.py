import os

PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "password")
PG_DATABASE = os.getenv("PG_DATABASE", "localhost")
PG_HOST = os.getenv("PG_HOST", "postgres")
PG_PORT = int(os.getenv("PG_PORT", 5432))

pg_dsn = f"dbname={PG_DATABASE} user={PG_USER} password={PG_PASSWORD} host={PG_HOST}"

AMQP_USER = os.getenv("AMQP_USER", "guest")
AMQP_PASSWORD = os.getenv("AMQP_PASSWORD", "guest")
AMQP_HOST = os.getenv("AMQP_HOST", "localhost")
AMQP_PREDICTION_QUEUE = os.getenv("AMQP_PREDICTION_QUEUE", "prediction")

