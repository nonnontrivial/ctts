import torch
import h3
from fastapi import APIRouter, HTTPException, Query
from typing import List
from pathlib import Path
from astroplan import Observer
from astropy.time import Time
from ..internal.model import NN, path_to_state_dict
from ..internal.cell import Cell
from ..internal.region import get_is_night_across_cells


router = APIRouter()

model = NN()
model.load_state_dict(torch.load(path_to_state_dict))
model.eval()


def create_cell_feature_vector(lat: float, lon: float) -> list:
    cell = Cell(utc_time=Time.now(), coords=(lat, lon))
    return [
        lat,
        lon,
        cell.elevation,
        cell.cloud_cover,
        cell.time_hour,
        *cell.moon_position,
    ]


@router.post("/infer", tags=["inference"])
async def infer(cell_ids: List[str], get_is_night: bool = Query(True)):
    torch.set_printoptions(sci_mode=False)
    try:
        start_time = Time.now()
        coords = [h3.cell_to_latlng(x) for x in cell_ids]
        x = torch.tensor(
            [[create_cell_feature_vector(*x) for x in coords]],
            dtype=torch.float32,
        )
        with torch.no_grad():
            inferred = model(x)
        end_time = Time.now()
        return {
            "generated_in": (end_time - start_time).to_value("ms"),
            "completed_at": end_time.iso,
            "units": {"inferred_brightnesses": "mpsas", "generated_in": "ms"},
            "inferred_brightnesses": {
                x: y[0] for x, y in zip(cell_ids, inferred.tolist()[0])
            },
        } | (
            {"is_night": get_is_night_across_cells(cell_ids, end_time)}
            if get_is_night
            else {}
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/version", tags=["version"])
async def version():
    import json

    model_metadata = Path(__file__).parent.parent / "internal" / "model.json"
    return json.loads(model_metadata.read_text())
