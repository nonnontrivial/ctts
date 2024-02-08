import pdb
from pathlib import Path
import typing as t

import numpy as np
import httpx
import pandas as pd

from .constants import HOURS_IN_DAY
from .stations import Station, known_stations
from ..pollution.utils import get_luminance_for_color_channels, to_linear
from ..pollution.pollution import ArtificialNightSkyBrightnessMapImage, Coords

gan_mn_dir = Path.cwd() / "data" / "gan_mn"
gan_mn_dataframe_filename = "gan_mn.csv"

ansb_map_image = ArtificialNightSkyBrightnessMapImage()


class GaNMNData:
    # columns that we will not be able to build up at runtime
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
        print(f"preparing to process {len(dfs)} dataframes..")
        df = pd.concat(dfs, ignore_index=True)
        print(f"sanitizing..")
        df = self._sanitize_df(df.iloc[:100])
        print(f"encoding dates..")
        df = self._encode_dates_in_df(df)
        print(f"applying coordinate data to stations..")
        df["lat"] = df.apply(self._get_lat_at_row, axis=1)
        df["lon"] = df.apply(self._get_lon_at_row, axis=1)
        print(f"applying ansb..")
        df["ansb"] = df.apply(self._get_artificial_light_pollution_at_row, axis=1)
        print(f"applying cloud cover..")
        df["cloud"] = df.apply(self._get_cloud_cover_at_row, axis=1)
        print(f"dropping nonconstructable columns..")
        df = df.drop(columns=self.nonconstructable_columns)
        self.df = df
        self.save_path = dataset_path.parent

    def _sanitize_df(self, gan_mn_frame: pd.DataFrame) -> pd.DataFrame:
        # see "What is the range of the Sky Quality Meters" section of http://www.unihedron.com/projects/darksky/faqsqm.php
        df = gan_mn_frame[gan_mn_frame["nsb"] > 7.00]
        df = gan_mn_frame[gan_mn_frame["nsb"] < 23.0]
        df = gan_mn_frame[gan_mn_frame["device_code"].isin(known_stations)]
        return df.reset_index()

    def _encode_dates_in_df(self, gan_mn_frame: pd.DataFrame) -> pd.DataFrame:
        # see https://www.kaggle.com/code/avanwyk/encoding-cyclical-features-for-deep-learning
        df = gan_mn_frame
        df["received_utc"] = pd.to_datetime(df["received_utc"])
        df["hour_sin"] = np.sin(2*np.pi*df["received_utc"].dt.hour/HOURS_IN_DAY)
        df["hour_cos"] = np.cos(2*np.pi*df["received_utc"].dt.hour/HOURS_IN_DAY)
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
        return station.get_cloud_cover(utc)

    def _get_artificial_light_pollution_at_row(self, row: pd.Series):
        lat, lon = row["lat"], row["lon"]
        r, g, b, _ = ansb_map_image.get_pixel_values_at_coords(Coords(float(lat), float(lon)))
        vR, vG, vB = r/255, g/255, b/255
        luminance = get_luminance_for_color_channels(vR, vG, vB)
        return luminance


    @property
    def correlations(self):
        df = self.df.select_dtypes(include=["int64", "float64"])
        return df.corr()

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
    except KeyboardInterrupt as e:
        print(f"\npress ctrl-c again to exit..")
    else:
        print(f"writing file at {gan_mn_data.save_path / gan_mn_dataframe_filename} ..")
        gan_mn_data.write_to_disk()
        print(f"correlations were:\n{gan_mn_data.correlations}\n\non dataframe\n{gan_mn_data.df.info}")
