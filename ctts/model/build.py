from pathlib import Path
from configparser import ConfigParser

import astropy.units as u
import numpy as np
import pandas as pd
from astroplan import Observer
from astropy.coordinates import EarthLocation
from astropy.time import Time

config = ConfigParser()
config.read("config.ini")

max_sqm = config.getint("sqm", "max")
min_sqm = config.getint("sqm", "min")

csv_filename = config.get("csv", "filename")

path_to_data_dir = Path(__file__).parent.parent.parent / "data"
path_to_preprocessed_csvs = path_to_data_dir / "globe_at_night"
csvs = path_to_preprocessed_csvs.glob("*.csv")

dataframes = [
    pd.read_csv(path, on_bad_lines="skip")
    for path in csvs
]

columns_to_drop = ["ID", "SQMSerial", "Constellation", "SkyComment", "LocationComment", "Country"]
columns_that_must_not_be_na = ["ObsDateTime", "Latitude", "Longitude", "Elevation", "CloudCover", "SQMReading"]

dataframes = [df for df in dataframes if all(c in df.columns for c in columns_to_drop + columns_that_must_not_be_na)]
df = pd.concat(dataframes, ignore_index=True)

df = df.drop(columns=columns_to_drop)
df = df.dropna(subset=columns_that_must_not_be_na, how="any", axis=0)

print("dropping rows outside of sqm range")
df = df[df["SQMReading"] <= max_sqm]
df = df[df["SQMReading"] >= min_sqm]
df = df.reset_index()

print(f"building datetime mapping")
# create utdatetime column in order to form sine-mapped uttimehour
df["UTDatetime"] = pd.to_datetime(df["ObsDateTime"], utc=True)
df["UTTimeHour"] = np.sin(2 * np.pi * df["UTDatetime"].dt.hour / 24)


def get_moon_altaz(datetime, lat, lon):
    """get moon position in altitude/azimuth"""
    time = Time(datetime)
    location = EarthLocation.from_geodetic(lon * u.degree, lat * u.degree)
    observer = Observer(location=location)
    return observer.moon_altaz(time)


class RowOps:
    @staticmethod
    def get_moon_alt_for_row(row: pd.Series):
        datetime = row["UTDatetime"]
        lat, lon = row["Latitude"], row["Longitude"]
        altaz = get_moon_altaz(datetime, lat, lon)
        alt = altaz.alt.value
        print(f"got moon altitude {alt} for row {row.name}")
        return alt

    @staticmethod
    def get_moon_az_for_row(row: pd.Series):
        datetime = row["UTDatetime"]
        lat, lon = row["Latitude"], row["Longitude"]
        altaz = get_moon_altaz(datetime, lat, lon)
        az = altaz.az.value
        print(f"got moon azimuth {az} for row {row.name}")
        return az


print(f"applying moon data to {df.shape[0]} rows")
df["MoonAlt"] = df.apply(RowOps.get_moon_alt_for_row, axis=1)
df["MoonAz"] = df.apply(RowOps.get_moon_az_for_row, axis=1)


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


print(f"mapping cloud cover to {df.shape[0]} rows")
df["CloudCover"] = df["CloudCover"].map(get_oktas_from_description)

print(f"writing file to {path_to_data_dir.as_posix()}")
df.to_csv(path_to_data_dir / csv_filename, index=False)
