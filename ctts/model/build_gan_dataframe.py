from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astroplan import Observer
from astropy.coordinates import EarthLocation
from astropy.time import Time

SQM_OBS_TYPE = "SQM"
MAX_SQM = 22
MIN_SQM = 16

cwd = Path.cwd()
path_tp_preprocessed_csvs = cwd / "data" / "globe_at_night"
dataframes = [
    pd.read_csv(p, on_bad_lines="skip")
    for p in path_tp_preprocessed_csvs.glob("*.csv")
    if p.stem != "GaN2022"
]
df = pd.concat(dataframes, ignore_index=True)
df = df.drop(columns=["ID", "ObsID", "LocalDate", "LocalTime", "Constellation"])
df = df.dropna(subset=["SQMReading", "CloudCover", "Elevation(m)"], how="any", axis=0)
df = df[df["ObsType"] == SQM_OBS_TYPE]
df = df[df["SQMReading"] <= MAX_SQM]
df = df[df["SQMReading"] >= MIN_SQM]
df = df.reset_index()
df["UTDatetime"] = pd.to_datetime(
    df["UTDate"] + " " + df["UTTime"], format="%Y-%m-%d %H:%M"
)
df["UTTimeHour"] = np.sin(2 * np.pi * df["UTDatetime"].dt.hour / 24)


def get_moon_altaz(datetime, lat, lon):
    time = Time(datetime)
    location = EarthLocation.from_geodetic(lon * u.degree, lat * u.degree)
    observer = Observer(location=location)
    return observer.moon_altaz(time)


def get_moon_alt(datetime, lat, lon):
    altaz = get_moon_altaz(datetime, lat, lon)
    return altaz.alt.value


def get_moon_az(datetime, lat, lon):
    altaz = get_moon_altaz(datetime, lat, lon)
    return altaz.az.value


df["MoonAlt"] = df.apply(
    lambda x: get_moon_alt(x["UTDatetime"], x["Latitude"], x["Longitude"]), axis=1
)
df["MoonAz"] = df.apply(
    lambda x: get_moon_az(x["UTDatetime"], x["Latitude"], x["Longitude"]), axis=1
)


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


df["CloudCover"] = df["CloudCover"].map(get_oktas_from_description)

if __name__ == "__main__":
    output_file_path = cwd / "data" / "globe_at_night.csv"
    df.to_csv(output_file_path, index=False)
