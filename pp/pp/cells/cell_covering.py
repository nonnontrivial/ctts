import json
import typing
from pathlib import Path

import h3
from shapely.geometry import shape, Polygon

RESOULTION = 0

def get_cell_id(lat, lon, resolution) -> str:
    return h3.geo_to_h3(lat, lon, resolution=resolution)


class H3CellCovering:
    """the cell covering that the publisher should publish over"""
    def __init__(self):
        self.resolution = RESOULTION
        self.polygon = self._ingest_polygon()

    def __call__(self, *args, **kwargs) -> typing.Set:
        return h3.polyfill_geojson(self.polygon, res=self.resolution)

    def _ingest_polygon(self) -> typing.Dict:
        path_to_geojson = Path(__file__).parent / f"map.geojson"
        with open(path_to_geojson, "r") as file:
            geojson = json.load(file)

        geojson_polygon = geojson["features"][0]["geometry"]
        polygon = shape(geojson_polygon)

        if not isinstance(polygon, Polygon):
            raise ValueError("could not parse geojson as polygon")

        return {
            "type": "Polygon",
            "coordinates": [list(polygon.exterior.coords)]
        }
