# FIXME move
features = [
    "Latitude",
    "Longitude",
    "Elevation",
    "CloudCover",
    "UTTimeHour",
    "MoonAlt",
    "MoonAz",
]
num_features = len(features)

HIDDEN_SIZE = 64 * 3
OUTPUT_SIZE = 1

MODEL_STATE_DICT_FILE_NAME = "model.pth"
MAX_OKTAS = 8

LOGFILE_KEY = "SKY_BRIGHTNESS_LOGFILE"
