from dataclasses import dataclass

@dataclass
class Coords:
    lat: float
    lon: float


@dataclass
class Pixel:
    x: int
    y: int
