import torch
import h3
from fastapi import APIRouter, HTTPException
from typing import List
from astroplan import Observer
from astropy.time import Time
from ..internal.model import NN, path_to_state_dict
from ..internal.cell import Cell


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
async def infer(cell_ids: List[str]):
    torch.set_printoptions(sci_mode=False)
    try:
        coords = [h3.cell_to_latlng(x) for x in cell_ids]
        x = torch.tensor(
            [[create_cell_feature_vector(*x) for x in coords]],
            dtype=torch.float32,
        )
        with torch.no_grad():
            inferred = model(x)
        end = Time.now()
        return {
            "times": {
                "completed_at": end.iso,
            },
            "units": {"inferred_brightnesses": "mpsas"},
            "inferred_brightnesses": {
                x: y[0] for x, y in zip(cell_ids, inferred.tolist()[0])
            },
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
