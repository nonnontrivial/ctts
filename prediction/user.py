import typing as t
from dataclasses import dataclass


@dataclass
class User:
    id: str
    site_ids: t.List[str]
    site_thresholds: t.Dict[str, float]
