import json
import typing
from pathlib import Path

import h3
from shapely.geometry import shape, Polygon

Continents = typing.Literal["north-america"]


class H3ContinentManager:
    def __init__(self, continent: Continents, resolution=6):
        self.continent = continent
        self.polygon = self._ingest_polygon()
        self.resolution = resolution

    def get_cell_covering(self) -> typing.Set:
        return h3.polyfill_geojson(self.polygon, res=self.resolution)

    def get_cell_id(self, lat, lon) -> str:
        return h3.geo_to_h3(lat, lon, resolution=self.resolution)

    def _ingest_polygon(self) -> typing.Dict:
        geojson_file_path = Path(__file__).parent / f"{self.continent}.geojson"
        with open(geojson_file_path, "r") as file:
            geojson = json.load(file)
        geojson_polygon = geojson["features"][0]["geometry"]
        polygon = shape(geojson_polygon)

        if not isinstance(polygon, Polygon):
            raise ValueError("could not parse geojson as polygon")

        return {
            "type": "Polygon",
            "coordinates": [list(polygon.exterior.coords)]
        }
