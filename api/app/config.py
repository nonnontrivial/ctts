import os

open_meteo_host = os.getenv("OPEN_METEO_HOST", "localhost")
db_path = os.getenv("DB_PATH", "../data/ctts.db")
