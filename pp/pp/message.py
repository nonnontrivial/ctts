from dataclasses import dataclass


@dataclass
class PredictionMessage:
    time_of: str
    lat: float
    lon: float
    sky_brightness_mpsas: float
