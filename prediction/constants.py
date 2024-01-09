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

SITE_NAME = "user-site"
LOGFILE_KEY = "SKY_BRIGHTNESS_LOGFILE"

API_PREFIX = "/api/v1"
