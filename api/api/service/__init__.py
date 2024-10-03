import tomllib
from pathlib import Path

with open(Path(__file__).parent / "config.toml", "rb") as file:
    config = tomllib.load(file)
