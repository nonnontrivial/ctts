# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "astroplan",
#     "astropy",
#     "numpy",
#     "pandas",
#     "requests>=2.32.3",
# ]
# ///

import logging
import typing
import tomllib
import pdb
from argparse import ArgumentParser
from pathlib import Path
from typing import Callable

import requests
import numpy as np
import pandas as pd
from astroplan import Observer
from astropy.coordinates import EarthLocation
from astropy.time import Time
import astropy.units as u


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

with open("config.toml", "rb") as f:
    config = tomllib.load(f)

obsolete_columns = config["data"]["obsolete_columns"]
required_columns = config["data"]["required_columns"]
min_sqm = config["data"]["min_sqm"]
max_sqm = config["data"]["max_sqm"]

open_meteo_host = config["data"]["open_meteo"]["host"]
open_meteo_port = config["data"]["open_meteo"]["port"]


class GaNFrame:
    def __init__(self, data_path: Path):
        self._df: pd.DataFrame | None = None
        self._data_path = data_path
        if not self._data_path.exists():
            raise FileNotFoundError()
        (self._data_path / "gan.csv").unlink(missing_ok=True)

    @property
    def df(self):
        return self._df

    def export(self) -> None:
        if self._df is not None:
            self._df.to_csv(self._data_path / "gan.csv", index=False)

    def load(self) -> None:
        log.info(f"loading data at {self._data_path.resolve()}..")
        dfs = [
            pd.read_csv(p, on_bad_lines="skip") for p in self._data_path.rglob("*.csv")
        ]
        dfs = [
            df.drop(columns=obsolete_columns, errors="ignore").dropna(
                subset=required_columns, how="any", axis=0
            )
            for df in dfs
            if all(c in df.columns for c in required_columns)
        ]
        df = pd.concat(dfs, ignore_index=True)
        df = df[df["SQMReading"] <= max_sqm]
        df = df[df["SQMReading"] >= min_sqm]
        df = df.reset_index()
        log.info(f"loaded {len(df)} rows from {len(dfs)} distinct dataframes")
        self._df = df

    def add_column(self, name: str, func: Callable[[pd.DataFrame], pd.Series]) -> None:
        if self._df is None:
            raise RuntimeError()
        log.info(f"adding column {name}")
        self._df[name] = self._df.apply(func, axis=1)

    def map_column(self, name: str, func: Callable[[pd.Series], pd.Series]) -> None:
        if self._df is None:
            raise RuntimeError()
        log.info(f"mapping column {name}")
        self._df[name] = self._df[name].map(func)


def confirm_overwrite(data_dir_path: Path) -> bool:
    path = data_dir_path / "gan.csv"
    res = input(f"this will remove {path.resolve()}\ncontinue? [y/N] ").strip().lower()
    return res in {"y", ""}


def get_moon_altaz(timestamp, lat: str, lon: str):
    time = Time(timestamp)
    location = EarthLocation.from_geodetic(lon * u.degree, lat * u.degree)
    observer = Observer(location=location)
    return observer.moon_altaz(time)


def get_temperature(timestamp, lat, lon):
    # https://historical-forecast-api.open-meteo.com/v1/forecast?latitude=52.52&longitude=13.41&start_date=2025-03-28&end_date=2025-04-11&hourly=temperature_2m
    res = requests.get(
        f"http://{open_meteo_host}:{open_meteo_port}/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": "",
            "end_date": "",
            "hourly": ["temperature_2m"],
        },
    )
    res.raise_for_status()
    data = res.json()
    df = pd.DataFrame(
        {
            "time": data["hourly"]["time"],
            "temperature_2m": data["hourly"]["temperature_2m"],
        }
    )
    df["time"] = pd.to_datetime(df["time"])
    target = timestamp
    closest_row = df.iloc[(df["time"] - target).abs().argsort().iloc[0]]
    # TODO ..
    return 0.0


def get_oktas_from_description(description: str) -> int:
    match description:
        case "0" | "clear":
            return 0
        case "25" | "1/4 of sky":
            return 2
        case "50" | "1/2 of sky":
            return 4
        case "75" | "over 1/2 of sky":
            return 6
        case _:
            return 8


def main(data_dir_path: Path) -> None:
    def get_days_since_oldest_row(df: pd.DataFrame) -> int:
        from datetime import timezone

        oldest_row = df.loc[df["ObsDateTime"].idxmin()]
        t0 = pd.to_datetime(oldest_row["ObsDateTime"], utc=True)
        now = pd.Timestamp.now(tz=timezone.utc)
        return (now - t0).days

    def store_historical_open_meteo_data_in_volume(
        gdf: pd.DataFrame, volume_name="ctts_open-meteo-data"
    ) -> None:
        import subprocess

        image = "ghcr.io/open-meteo/open-meteo"
        days_ago = get_days_since_oldest_row(gdf)
        cmd = f"docker run -it --rm -v {volume_name}:/app/data {image} sync copernicus_era5_land temperature_2m --past-days {days_ago}"
        try:
            res = subprocess.run(cmd, shell=True, check=True)
            log.info(f"syncing open meteo (from {days_ago}d)")
        except subprocess.CalledProcessError as e:
            log.error(f"could not sync open meteo: {e.returncode}")

    if not data_dir_path.exists():
        raise FileNotFoundError(f"{data_dir_path} does not exist")
    if not confirm_overwrite(data_dir_path):
        raise ValueError("cancelled by user")

    gan_frame = GaNFrame(data_path=data_dir_path)
    gan_frame.load()

    # store_historical_open_meteo_data_in_volume(gan_frame.df)
    gan_frame.add_column(
        "UTDatetime", lambda row: pd.to_datetime(row["ObsDateTime"], utc=True)
    )
    gan_frame.add_column(
        "UTTimeHour", lambda row: np.sin(2 * np.pi * row["UTDatetime"].hour / 24)
    )
    # gan_frame.add_column(
    #     "Temperature",
    #     lambda row: get_temperature(
    #         row["UTDatetime"], row["Latitude"], row["Longitude"]
    #     ),
    # )
    gan_frame.add_column(
        "MoonAlt",
        lambda row: get_moon_altaz(
            row["UTDatetime"], row["Latitude"], row["Longitude"]
        ).alt.value,
    )
    gan_frame.add_column(
        "MoonAz",
        lambda row: get_moon_altaz(
            row["UTDatetime"], row["Latitude"], row["Longitude"]
        ).az.value,
    )
    gan_frame.map_column("CloudCover", get_oktas_from_description)
    gan_frame.export()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data_dir_path", type=Path)
    try:
        main(**vars(parser.parse_args()))
    except KeyboardInterrupt as _:
        pass

