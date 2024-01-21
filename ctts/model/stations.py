from datetime import datetime
from dataclasses import dataclass

@dataclass
class Station:
    lat: float
    lon: float
    code: str
    organization: str
    date_of_first_operation: datetime

# see http://globeatnight-network.org/global-at-night-monitoring-network.html
known_stations = {
    "MP": (22.4949859, 114.032692),
    "Zwi": (48.99929, 13.21652)
}

def get_device_code_is_known_station(device_code: str):
    return device_code in known_stations
