import typing
from pathlib import Path

from PIL import Image

from . import config
from .models import Coords, Pixel


default_path_to_map_image = Path(__file__).parent.parent.parent / "data" / "map" / config["image"]["filename"]


class PollutionImage:
    def __init__(self, path_to_map_image: Path = default_path_to_map_image) -> None:

        if not path_to_map_image.exists():
            raise FileNotFoundError(f"{path_to_map_image} does not exist")

        image_mode = config["image"]["mode"]
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

    def _get_pixel_from_coords(self, coords: Coords) -> Pixel:
        # see https://gis.stackexchange.com/a/372118
        width_scale = self.image.width / self.max_lon_degs
        height_scale = self.image.height / self.max_lat_degs
        x = int((coords.lon + 180) * width_scale)
        y = int((self.max_lat_n - coords.lat) * height_scale)
        return Pixel(x, y)

    def _get_pixel_value(self, pixel: Pixel) -> typing.Tuple[int, int, int, int]:
        try:
            x, y = (pixel.x, pixel.y)
            return self.image.getpixel((x,y)) # type: ignore
        except IndexError:
            return 0, 0, 0, 255

    def get_rgba_at_coords(self, lat:float, lon:float) -> typing.Tuple[int,int,int,int]:
        """determine what is the RGBA value in the image at coords"""
        # see https://djlorenz.github.io/astronomy/lp2022/colors.html
        coords = Coords(lat=lat,lon=lon)
        pixel = self._get_pixel_from_coords(coords)
        pixel_value = self._get_pixel_value(pixel)
        return pixel_value
