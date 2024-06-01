from dataclasses import dataclass


@dataclass
class PredictionResponse:
    """in magnitudes per square arcsecond"""
    sky_brightness: float
