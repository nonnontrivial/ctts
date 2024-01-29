from dataclasses import dataclass

# see https://stackoverflow.com/a/56678483
def to_linear(color_channel_value: float) -> float:
    return color_channel_value / 12.92 if color_channel_value <= 0.04045 else ((color_channel_value+0.055)/1.055)**2.4

def get_luminance_for_color_channels(vR, vG, vB: float) -> float:
    return 0.2126 * to_linear(vR) + 0.7152 * to_linear(vG) + 0.0722 * to_linear(vB)

@dataclass
class Coords:
    lat: float
    lon: float
