from pathlib import Path
from enum import Enum
import typing as t

import numpy as np
import httpx
import pandas as pd

from .constants import HOURS_IN_DAY
from .stations import Station, known_stations

from ..pollution.utils import get_luminance_for_color_channels
from ..pollution.pollution import ArtificialNightSkyBrightnessMapImage, Coords

gan_mn_dir = Path.cwd() / "data" / "gan_mn"
gan_mn_dataframe_filename = "gan_mn.csv"

ansb_map_image = ArtificialNightSkyBrightnessMapImage()


class Features(Enum):
    HOUR_SIN = "hour_sin"
    HOUR_COS = "hour_cos"
    LAT = "lat"
    LON = "lon"
    ANSB = "ansb"
    CLOUD = "cloud"
    TEMP = "temperature"
    ELEV = "elevation"


class GaNMNData:
    """Dataframe of the Globe at Night Monitoring Network dataset.

    Includes methods to supplement the hosted data with columns from open meteo,
    in order to improve model.
    """
    # temporary row limit
    row_limit = 100000

    # The columns that we will not be able to build up at runtime.
    # `temperature` is reconstructed with the result from open meteo.
    nonconstructable_columns = [
        "id",
        "created",
        "received_adjusted",
        "sqmle_serial_number",
        "sensor_frequency",
        "sensor_period_count",
        "sensor_period_second",
        "device_code",
    ]

    def __init__(self, dataset_path: Path) -> None:
        dfs = [pd.read_csv(f) for f in dataset_path.glob("*.csv")]
        print(f"preparing to process {len(dfs)} dataframe(s)..")
        df = pd.concat(dfs, ignore_index=True)
        df = df.iloc[:self.row_limit]
        print(f"sanitizing dataframe with {len(df)} rows..")

        df = self._sanitize_df(df)
        print(f"encoding dates for {len(df)} rows..")
        df = self._encode_dates_in_df(df)

        print("applying coordinate data to stations..")
        df[Features.LAT.value] = df.apply(self._get_lat_at_row, axis=1)
        df[Features.LON.value] = df.apply(self._get_lon_at_row, axis=1)
        print("applying ansb..")
        df[Features.ANSB.value] = df.apply(
            self._get_artificial_light_pollution_at_row, axis=1
        )
        print("applying cloud cover..")
        df[Features.CLOUD.value] = df.apply(self._get_cloud_cover_at_row, axis=1)
        print("applying temperature..")
        df[Features.TEMP.value] = df.apply(self._get_temperature_at_row, axis=1)
        print("applying elevation..")
        df[Features.ELEV.value] = df.apply(self._get_elevation_at_row, axis=1)

        print("dropping nonconstructable columns..")
        df = df.drop(columns=self.nonconstructable_columns)

        self.df = df
        self.save_path = dataset_path.parent

    def _sanitize_df(self, gan_frame: pd.DataFrame) -> pd.DataFrame:
        # see "What is the range of the Sky Quality Meters" section of http://www.unihedron.com/projects/darksky/faqsqm.php
        df: t.Any = gan_frame[gan_frame["nsb"] > 7.0]
        df = df[df["device_code"].isin(known_stations)]
        return df.reset_index()

    def _encode_dates_in_df(self, gan_mn_frame: pd.DataFrame) -> pd.DataFrame:
        # see https://www.kaggle.com/code/avanwyk/encoding-cyclical-features-for-deep-learning
        df = gan_mn_frame
        df["received_utc"] = pd.to_datetime(df["received_utc"])
        df[Features.HOUR_SIN.value] = np.sin(
            2 * np.pi * df["received_utc"].dt.hour / HOURS_IN_DAY
        )
        df[Features.HOUR_COS.value] = np.cos(
            2 * np.pi * df["received_utc"].dt.hour / HOURS_IN_DAY
        )
        return df.reset_index()

    def _get_lat_at_row(self, row: pd.Series):
        device_code = str(row["device_code"])
        return known_stations[device_code][0]

    def _get_lon_at_row(self, row: pd.Series):
        device_code = str(row["device_code"])
        return known_stations[device_code][1]

    def _get_cloud_cover_at_row(self, row: pd.Series):
        device_code: t.Any = row["device_code"]
        station = Station(device_code)
        utc: t.Any = row["received_utc"]
        try:
            cloud_cover = station.get_cloud_cover(utc)
            print(f"cloud cover at {station} was {cloud_cover} ({row.name})")
            return cloud_cover
        except (httpx.ReadTimeout, httpx.ConnectError) as e:
            print(f"timed out when attempting to get cloud cover: {e}")
            return 0.0
        except Exception as e:
            print(f"could not get cloud cover: {e}")
            return 0.0

    def _get_temperature_at_row(self, row: pd.Series):
        device_code: t.Any = row["device_code"]
        station = Station(device_code)
        utc: t.Any = row["received_utc"]
        try:
            temperature = station.get_temperature(utc)
            print(f"temperature at {station} was {temperature}")
            return temperature
        except (httpx.ReadTimeout, httpx.ConnectError) as e:
            print(f"timed out when attempting to get temperature: {e}")
            return 0.0
        except Exception as e:
            print(f"could not get temperature: {e}")
            return 0.0

    def _get_elevation_at_row(self, row: pd.Series):
        device_code: t.Any = row["device_code"]
        station = Station(device_code)
        try:
            elevation = station.elevation
            print(f"elevation at {station} is {elevation}")
            return elevation
        except Exception:
            print(f"failed to apply elevation to {station}")
            return 0.0

    def _get_artificial_light_pollution_at_row(self, row: pd.Series):
        lat, lon = row[Features.LAT.value], row[Features.LON.value]
        r, g, b, _ = ansb_map_image.get_pixel_values_at_coords(
            Coords(float(lat), float(lon))
        )
        vR, vG, vB = r / 255, g / 255, b / 255
        luminance = get_luminance_for_color_channels(vR, vG, vB)
        return luminance

    @property
    def correlations(self):
        return self.df.corr()

    def write_to_disk(self) -> None:
        self.df.to_csv(self.save_path / gan_mn_dataframe_filename, index=False)


if __name__ == "__main__":
    if not gan_mn_dir.exists():
        raise FileNotFoundError(f"!missing {gan_mn_dir}")
    print(f"loading dataset at {gan_mn_dir} ..")
    try:
        gan_mn_data = GaNMNData(gan_mn_dir)
    except ValueError as e:
        print(f"!failed to create dataframe: {e}")
    except KeyboardInterrupt:
        print("\npress ctrl-c again to exit..")
    else:
        print(f"writing file at {gan_mn_data.save_path / gan_mn_dataframe_filename} ..")
        gan_mn_data.write_to_disk()
        info = gan_mn_data.df.head()
        print(f"correlations were:\n{gan_mn_data.correlations}\n\non dataframe\n{info}")
