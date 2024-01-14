import pdb
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

@dataclass
class Coords:
    lat: float
    lon: float

@dataclass
class Pixel:
    x: int
    y: int

class WorldMapImage:
    def __init__(self, path_to_map_image: Path, mode = "RGBA") -> None:
        if not path_to_map_image.exists():
            raise FileNotFoundError(f"{path_to_map_image} does not exist")
        self.image = Image.open(path_to_map_image).convert(mode)

    def get_pixel_in_image_from_coords(self,coords:Coords) -> Pixel:
        width_scale = self.image.width / 360
        height_scale = self.image.height / 180
        x = int((lon+180) * width_scale)
        y = int((90-lat) * height_scale)
        pdb.set_trace()
        # size = 100
        # cropped = self.image.crop((x-size,y-size,x+size,y+size))
        # cropped.save("crop.png")
        return Pixel(x,y)

    def get_pixel_value(self,pixel:Pixel) -> int:
        return self.image.getpixel((pixel.x, pixel.y))

class ArtificialSkyBrightnessMapImage(WorldMapImage):
    def get_mpsas_range_at_coords(self, coords:Coords) -> int:
        pixel = self.get_pixel_in_image_from_coords(coords)
        pixel_value = self.get_pixel_value(pixel)
        return pixel_value

if __name__ == "__main__":
    (lat, lon) = (40.730610, -73.935242)
    path_to_image = Path.cwd().parent.parent / "data" / "artificial_night_sky_brightness" / "world2022B.png"
    asbm = ArtificialSkyBrightnessMapImage(path_to_image)
    mr = asbm.get_mpsas_range_at_coords(coords=Coords(lat=lat,lon=lon))
    print(mr)
