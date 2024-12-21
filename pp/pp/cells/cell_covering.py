import json
import typing
from pathlib import Path

import h3
from shapely.geometry import shape, Polygon

from ..config import resolution

class CellCovering:
    def __init__(self, path_to_geojson: Path | None = None):
        if path_to_geojson is None:
            path_to_geojson = Path(__file__).parent / "land.geojson"

        with open(path_to_geojson, "r") as file:
            geojson = json.load(file)

        self.polygons = [CellCovering.get_polygon_of_feature(f) for f in geojson["features"]]

    @staticmethod
    def get_cell_id(lat, lon, resolution) -> str:
        return h3.geo_to_h3(lat, lon, resolution=resolution)


    @staticmethod
    def get_polygon_of_feature(feature: typing.Dict) -> typing.Dict:
        polygon = shape(feature["geometry"])
        if not isinstance(polygon, Polygon):
            raise TypeError("geojson is not a Polygon")

        return {
            "type": "Polygon",
            "coordinates": [list(polygon.exterior.coords)]
        }

    @property
    def covering(self) -> typing.Set:
        """the complete covering from all features in the collection"""
        all_polygons=set().union(*[h3.polyfill_geojson(p, res=resolution) for p in self.polygons])
        return all_polygons
