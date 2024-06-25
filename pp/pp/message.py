from dataclasses import dataclass


@dataclass
class PredictionMessage:
    utc: str
    lat: float
    lon: float
    # magnitudes per square arc second
    mpsas: float
    # id of the h3 cell
    h3_id: str
