from typing import List, Tuple
import h3


def get_res_zero_cell_coords() -> List[Tuple[float, float]]:
    """gets list of coordinates of all resolution zero cells"""
    resolution_zero_cells = h3.get_res0_indexes()
    return [h3.h3_to_geo(c) for c in resolution_zero_cells]
