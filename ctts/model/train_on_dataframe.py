import pdb
from typing import Tuple,Any
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset,random_split

from .build_dataframe import gan_mn_dir, gan_mn_dataframe_filename

class GaNMNDataset(Dataset):
    features = [
        "temperature",
        "hour_sin",
        "hour_cos",
        "lat",
        "lon",
        "ansb",
    ]
    label = "nsb"
    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__()
        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Any,Any]:
        features = torch.tensor(self.data[self.features].values, dtype=torch.float32)
        label = torch.tensor(self.data[self.label], dtype=torch.float32)
        return features, label

def get_train_test_split_size(dataset_size: int) -> Tuple[int,int]:
    train_size = int(0.8*dataset_size)
    test_size = dataset_size - train_size
    return train_size,test_size

def train_model() -> None:
    path_to_dataframe_file = gan_mn_dir.parent / gan_mn_dataframe_filename
    df = pd.read_csv(path_to_dataframe_file)
    dataset = GaNMNDataset(df=df)

    train_dataset,test_dataset=random_split(dataset,get_train_test_split_size(len(dataset)))

    for X, y in train_dataset:
        pdb.set_trace()

if __name__ == "__main__":
    torch.set_printoptions(sci_mode=False)
    train_model()
