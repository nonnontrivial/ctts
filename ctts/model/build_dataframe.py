import pdb
from pathlib import Path
import typing as t

import numpy as np
import pandas as pd

from .stations import known_stations, get_device_code_is_known_station
from ..pollution.pollution import ArtificialNightSkyBrightnessMapImage, Coords

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

ansb_map_image = ArtificialNightSkyBrightnessMapImage()

class GaNMNData:
    output_filename = "gan_mn.csv"

    def __init__(self, data_path: Path) -> None:
        records = data_path.glob("*.csv")
        gan_mn_dataframes = [pd.read_csv(record, on_bad_lines="skip") for record in records]
        df = pd.concat(gan_mn_dataframes, ignore_index=True)
        df = self._sanitize_df(df)
        df = self._encode_dates_in_df(df)
        df["lat"] = df.apply(self._get_lat_at_row, axis=1)
        df["lon"] = df.apply(self._get_lon_at_row, axis=1)
        df["ansb"] = df.apply(self._get_artificial_light_pollution_at_row, axis=1)
        df = df.drop(columns=nonconstructable_columns)
        self.df = df
        self.save_path = data_path.parent

    def _sanitize_df(self, gan_mn_frame: pd.DataFrame) -> pd.DataFrame:
        df = gan_mn_frame[gan_mn_frame["nsb"] > 0.00]
        df = gan_mn_frame[gan_mn_frame["device_code"].isin(known_stations)]
        return df.reset_index()

    def _encode_dates_in_df(self, gan_mn_frame: pd.DataFrame) -> pd.DataFrame:
        # see https://www.kaggle.com/code/avanwyk/encoding-cyclical-features-for-deep-learning
        df = gan_mn_frame
        df["received_utc"] = pd.to_datetime(df["received_utc"])
        hours_in_day = 24.
        df["hour_sin"] = np.sin(2*np.pi*df["received_utc"].dt.hour/hours_in_day)
        df["hour_cos"] = np.cos(2*np.pi*df["received_utc"].dt.hour/hours_in_day)
        return df.reset_index()

    def _get_lat_at_row(self, row: pd.Series):
        device_code = str(row["device_code"])
        return known_stations[device_code][0]

    def _get_lon_at_row(self, row: pd.Series):
        device_code = str(row["device_code"])
        return known_stations[device_code][1]

    def _get_artificial_light_pollution_at_row(self, row: pd.Series):
        # see https://stackoverflow.com/a/56678483
        def to_linear(color_channel_value: float) -> float:
            return color_channel_value / 12.92 if color_channel_value <= 0.04045 else ((color_channel_value+0.055)/1.055)**2.4
        lat, lon = row["lat"], row["lon"]
        r, g, b, _ = ansb_map_image.get_pixel_value_at_coords(Coords(float(lat), float(lon)))
        vR, vG, vB = r/255, g/255, b/255
        luminance = 0.2126 * to_linear(vR) + 0.7152 * to_linear(vG) + 0.0722 * to_linear(vB)
        return luminance

    @property
    def correlations(self):
        df = self.df.select_dtypes(include=["int64", "float64"])
        return df.corr()

    def write_to_disk(self) -> None:
        self.df.to_csv(self.save_path / self.output_filename, index=False)

if __name__ == "__main__":
    data_path = Path.cwd() / "data" / "gan_mn"
    if not data_path.exists():
        raise FileNotFoundError(f"!missing {data_path}")
    print(f"loading dataset at {data_path} ..")
    try:
        gan_mn_network_data = GaNMNData(data_path)
    except ValueError as e:
        print(f"!failed to create dataframe: {e}")
    else:
        print(f"writing file at {gan_mn_network_data.save_path / gan_mn_network_data.output_filename} ..")
        gan_mn_network_data.write_to_disk()
        print(f"correlations were:\n{gan_mn_network_data.correlations}")
