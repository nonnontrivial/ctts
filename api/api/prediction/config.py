import os

open_meteo_host = os.getenv("OPEN_METEO_HOST", "localhost")
open_meteo_port = int(os.getenv("OPEN_METEO_PORT", "8080"))
