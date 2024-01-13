features = [
    "Latitude",
    "Longitude",
    "Elevation(m)",
    "CloudCover",
    "UTTimeHour",
    "MoonAlt",
    "MoonAz",
]

HIDDEN_SIZE = 64 * 3
OUTPUT_SIZE = 1
FEATURES_SIZE = len(features)

ASTRO_TWILIGHT_DEGS = -18

OPEN_METEO_BASE_URL = "https://api.open-meteo.com"

MODEL_STATE_DICT_FILE_NAME = "model.pth"
SITE_NAME = "user-site"
LOGFILE_KEY = "SKY_BRIGHTNESS_LOGFILE"
