from dataclasses import dataclass
from datetime import datetime


@dataclass
class Site:
    lat: str
    lon: str
    astro_twilight_datetime: datetime
