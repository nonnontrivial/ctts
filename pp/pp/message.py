from dataclasses import dataclass


@dataclass
class PredictionMessage:
    utc: str
    lat: float
    lon: float
    # magnitudes per square arc second
    mpsas: float
    h3_id: str
