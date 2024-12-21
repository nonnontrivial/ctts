import json
from pathlib import Path

import pytest
from pp.cells.cell_covering import CellCovering

@pytest.mark.parametrize("geojson_path", [
    (Path.cwd() / "pp" / "cells" / "land.geojson")
])
def test_cell_covering_polygons_is_one_to_one_with_features(geojson_path):
    cell_covering = CellCovering(path_to_geojson=geojson_path)
    with open(geojson_path) as f:
        gj = json.load(f)

    assert len(cell_covering.polygons) == len(gj["features"])

@pytest.mark.parametrize("geojson_path", [
    (Path.cwd() / "pp" / "cells" / "land.geojson")
])
def test_cell_covering_set_nonempty(geojson_path):
    cell_covering = CellCovering(path_to_geojson=geojson_path)
    assert bool(cell_covering.covering)

@pytest.mark.parametrize("geojson_path", [
    (Path.cwd() / "pp" / "cells" / "fake.geojson")
])
def test_cell_covering_with_bad_geojson_path(geojson_path):
    with pytest.raises(FileNotFoundError):
        cell_covering = CellCovering(path_to_geojson=geojson_path)
