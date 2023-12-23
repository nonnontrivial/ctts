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
TEST_LAT = 43.05148
TEST_LON = -78.57732
OPEN_METEO_BASE_URL = "https://api.open-meteo.com"
SITE_NAME = "some-h3-cell"
