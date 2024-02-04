# import pdb
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .utils import Coords

@dataclass
class Pixel:
    x: int
    y: int

image_mode = "RGBA"
default_path_to_map_image = Path.cwd() / "data" / "ansb" / "world2022.png"

class ArtificialNightSkyBrightnessMapImage:
    def __init__(self, path_to_map_image: Path = default_path_to_map_image) -> None:
        if not path_to_map_image.exists():
            raise FileNotFoundError(f"{path_to_map_image} does not exist")
        self.image = Image.open(path_to_map_image).convert(image_mode)
        # see domain column in table at https://djlorenz.github.io/astronomy/lp2022/
        self.max_lat_n = 75
        self.max_lat_s = 65

    @property
    def max_lat_degs(self):
        return self.max_lat_n + self.max_lat_s

    @property
    def max_lon_degs(self):
        return 180 * 2

    def get_pixel_in_image_from_coords(self, coords: Coords) -> Pixel:
        # see https://gis.stackexchange.com/a/372118
        width_scale = self.image.width / self.max_lon_degs
        height_scale = self.image.height / self.max_lat_degs
        x = int((coords.lon+180) * width_scale)
        y = int((self.max_lat_n-coords.lat) * height_scale)
        # pdb.set_trace()
        return Pixel(x, y)

    def get_pixel_value(self, pixel: Pixel) -> tuple[int,int,int,int]:
        try:
            return self.image.getpixel((pixel.x, pixel.y))
        except IndexError:
            return (0, 0, 0, 255)

    def get_pixel_values_at_coords(self, coords: Coords) -> tuple[int,int,int,int]:
        # see https://djlorenz.github.io/astronomy/lp2022/colors.html
        pixel = self.get_pixel_in_image_from_coords(coords)
        pixel_value = self.get_pixel_value(pixel)
        # pdb.set_trace()
        return pixel_value
