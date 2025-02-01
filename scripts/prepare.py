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

import pandas as pd

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
    import numpy as np

    # create utdatetime column in order to form sine-mapped uttimehour
    df["UTDatetime"] = pd.to_datetime(df["ObsDateTime"], utc=True)
    df["UTTimeHour"] = np.sin(2 * np.pi * df["UTDatetime"].dt.hour / 24)
    return df


def add_moon_columns(df: pd.DataFrame) -> pd.DataFrame:
    from astroplan import Observer
    from astropy.coordinates import EarthLocation
    from astropy.time import Time
    import astropy.units as u

    def get_moon_altaz(datetime, lat, lon):
        """get moon position in altitude/azimuth"""
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


def main() -> None:
    """collect the brightness data into a single dataframe, and then write it to
    csv file for model training."""
    data_dir_path = list(Path.cwd().parent.rglob("brightness_data"))[0]
    assert data_dir_path.exists(), "data dir path des not exist!"

    log.info("getting single dataframe from source files")
    df = get_single_dataframe(data_dir_path)
    assert df is not None, "no dataframe!"

    log.info("dropping unnecessary columns")
    df = df.drop(columns=columns_to_drop)
    df = df.dropna(subset=columns_that_must_not_be_na, how="any", axis=0)

    log.info("dropping rows outside of sqm range")
    df = df[df["SQMReading"] <= max_sqm]
    df = df[df["SQMReading"] >= min_sqm]
    df = df.reset_index()

    log.info("adding date columns")
    df = add_date_columns(df)

    log.info("adding moon columns")
    df = add_moon_columns(df)

    log.info("adding cloud cover columns")
    df = add_cloud_cover_columns(df)

    log.info(f"writing {csv_filename} to {data_dir_path}")
    df.to_csv(data_dir_path / csv_filename, index=False)


if __name__ == "__main__":
    main()
