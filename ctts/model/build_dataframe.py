import pdb
from pathlib import Path
import typing as t

import numpy as np
import pandas as pd

from .constants import HOURS_IN_DAY
from .stations import known_stations
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
        records = dataset_path.glob("*.csv")
        gan_mn_dataframes = [pd.read_csv(record, on_bad_lines="skip") for record in records]
        df = pd.concat(gan_mn_dataframes, ignore_index=True)
        df = self._sanitize_df(df)
        df = self._encode_dates_in_df(df)
        df["lat"] = df.apply(self._get_lat_at_row, axis=1)
        df["lon"] = df.apply(self._get_lon_at_row, axis=1)
        df["ansb"] = df.apply(self._get_artificial_light_pollution_at_row, axis=1)
        df = df.drop(columns=self.nonconstructable_columns)
        self.df = df
        self.save_path = dataset_path.parent

    def _sanitize_df(self, gan_mn_frame: pd.DataFrame) -> pd.DataFrame:
        df = gan_mn_frame[gan_mn_frame["nsb"] > 0.00]
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
    else:
        print(f"writing file at {gan_mn_data.save_path / gan_mn_dataframe_filename} ..")
        gan_mn_data.write_to_disk()
        print(f"correlations were:\n{gan_mn_data.correlations}\n\non dataframe\n{gan_mn_data.df.info}")
