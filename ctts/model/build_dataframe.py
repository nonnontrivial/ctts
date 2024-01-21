import pdb
from pathlib import Path
import typing as t

import numpy as np
import pandas as pd

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

class GaNMNData:
    output_filename = "gan_mn.csv"

    def __init__(self, data_path: Path) -> None:
        records = data_path.glob("*.csv")
        gan_mn_dataframes = [pd.read_csv(record, on_bad_lines="skip") for record in records]
        df = pd.concat(gan_mn_dataframes, ignore_index=True)
        df = self._sanitize_df(df)
        df = self._encode_dates_in_df(df)
        self.df = df
        self.save_path = data_path.parent

    def _sanitize_df(self, gan_mn_frame: pd.DataFrame) -> pd.DataFrame:
        df = gan_mn_frame[gan_mn_frame["nsb"] > 0.00]
        df = df.drop(columns=nonconstructable_columns)
        return df.reset_index()

    def _encode_dates_in_df(self, gan_mn_frame: pd.DataFrame) -> pd.DataFrame:
        # see https://www.kaggle.com/code/avanwyk/encoding-cyclical-features-for-deep-learning
        df = gan_mn_frame
        df["received_utc"] = pd.to_datetime(df["received_utc"])
        hours_in_day = 24.
        df["hour_sin"] = np.sin(2*np.pi*df["received_utc"].dt.hour/hours_in_day)
        df["hour_cos"] = np.cos(2*np.pi*df["received_utc"].dt.hour/hours_in_day)
        return df.reset_index()

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
        print(f"!are there .csv files in {data_path}?: {e}")
    else:
        print(f"writing file at {gan_mn_network_data.save_path / gan_mn_network_data.output_filename} ..")
        gan_mn_network_data.write_to_disk()
        print(f"correlations were:\n{gan_mn_network_data.correlations}")
