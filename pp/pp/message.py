from dataclasses import dataclass


@dataclass
class PredictionMessage:
    lat: float
    lon: float
    utc: str
    # magnitudes per square arc second
    mpsas: float
    # id of the h3 cell
    h3_id: str
