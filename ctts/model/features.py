from enum import Enum

class Features(Enum):
    """
    Columns to be used in feature vector
    """

    HOUR_SIN = "hour_sin"
    HOUR_COS = "hour_cos"
    LAT = "lat"
    LON = "lon"
    ANSB = "ansb"
    CLOUD = "cloud"
    TEMP = "temperature"
    ELEV = "elevation"
