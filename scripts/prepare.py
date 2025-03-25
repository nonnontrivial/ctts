# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "astroplan",
#     "astropy",
#     "numpy",
#     "pandas",
# ]
# ///

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from astroplan import Observer
from astropy.coordinates import EarthLocation
from astropy.time import Time
import astropy.units as u


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

csv_filename = "gan.csv"
min_sqm = 16
max_sqm = 22
columns_to_drop = [
    "ID",
    "SQMSerial",
    "Constellation",
    "SkyComment",
    "LocationComment",
    "Country",
]
columns_that_must_not_be_na = [
    "ObsDateTime",
    "Latitude",
    "Longitude",
    "Elevation",
    "CloudCover",
    "SQMReading",
]


def get_single_dataframe(data_dir_path: Path) -> pd.DataFrame:
    dfs = [
        pd.read_csv(path, on_bad_lines="skip") for path in data_dir_path.rglob("*.csv")
    ]
    dfs = [
        df
        for df in dfs
        if all(c in df.columns for c in columns_to_drop + columns_that_must_not_be_na)
    ]
    df = pd.concat(dfs, ignore_index=True)
    return df


def add_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    df["UTDatetime"] = pd.to_datetime(df["ObsDateTime"], utc=True)
    df["UTTimeHour"] = np.sin(2 * np.pi * df["UTDatetime"].dt.hour / 24)
    return df


def add_moon_columns(df: pd.DataFrame) -> pd.DataFrame:
    def get_moon_altaz(datetime, lat, lon):
        """get moon position (altitude, azimuth)"""
        time = Time(datetime)
        location = EarthLocation.from_geodetic(lon * u.degree, lat * u.degree)
        observer = Observer(location=location)
        return observer.moon_altaz(time)

    def get_moon_alt_for_row(row: pd.Series):
        datetime = row["UTDatetime"]
        lat, lon = row["Latitude"], row["Longitude"]
        altaz = get_moon_altaz(datetime, lat, lon)
        alt = altaz.alt.value
        return alt

    def get_moon_az_for_row(row: pd.Series):
        datetime = row["UTDatetime"]
        lat, lon = row["Latitude"], row["Longitude"]
        altaz = get_moon_altaz(datetime, lat, lon)
        az = altaz.az.value
        return az

    df["MoonAlt"] = df.apply(get_moon_alt_for_row, axis=1)
    df["MoonAz"] = df.apply(get_moon_az_for_row, axis=1)
    return df


def add_cloud_cover_columns(df: pd.DataFrame) -> pd.DataFrame:
    def get_oktas_from_description(description: str) -> int:
        """map description of cloud coverage into int"""
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

    df["CloudCover"] = df["CloudCover"].map(get_oktas_from_description)
    return df


def main(data_dir_path: Path) -> None:
    csv_path = data_dir_path / csv_filename
    if input(f"this will remove {csv_path}. Continue? [y/N] ").strip().lower() not in {
        "y",
        "",
    }:
        raise ValueError("User cancelled")
    csv_path.unlink(missing_ok=True)

    log.info("getting single dataframe from source files")
    df = get_single_dataframe(data_dir_path)
    if df is None:
        raise ValueError("no dataframe!")

    log.info("dropping unnecessary columns")
    df = df.drop(columns=columns_to_drop).dropna(
        subset=columns_that_must_not_be_na, how="any", axis=0
    )

    log.info("dropping rows outside of sqm range")
    df = df[df["SQMReading"] <= max_sqm]
    df = df[df["SQMReading"] >= min_sqm]
    df = df.reset_index()

    log.info(f"adding date columns to {len(df)} rows")
    df = add_date_columns(df)

    log.info(f"adding moon columns to {len(df)} rows")
    df = add_moon_columns(df)

    log.info(f"adding cloud cover columns to {len(df)} rows")
    df = add_cloud_cover_columns(df)

    log.info(f"writing {csv_path}")
    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    data_dir_path = Path.cwd() / "gan-data"
    main(data_dir_path)
