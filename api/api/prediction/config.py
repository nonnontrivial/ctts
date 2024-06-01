import os

open_meteo_host = os.getenv("OPEN_METEO_HOST", "localhost")
open_meteo_port = int(os.getenv("OPEN_METEO_PORT", "8080"))

model_state_dict_file_name = os.getenv("MODEL_STATE_DICT_FILE_NAME", "model.pth")
