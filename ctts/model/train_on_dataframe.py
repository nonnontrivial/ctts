import pdb
from dataclasses import dataclass
from typing import Tuple, Any, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset, TensorDataset,random_split

from .build_dataframe import gan_mn_dir, gan_mn_dataframe_filename
from .net import LinearNet

def get_train_test_split_size(dataset_size: int) -> Tuple[int,int]:
    train_size = int(0.8*dataset_size)
    test_size = dataset_size - train_size
    return train_size,test_size

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


label = "nsb"
features = [
    "temperature",
    "hour_sin",
    "hour_cos",
    "lat",
    "lon",
    "ansb",
]

def train_model(path: Path) -> Dict[str, Any]:
    print(f"loading dataset {path_to_dataframe_file}")
    df = pd.read_csv(path_to_dataframe_file)


    feature_tensor = torch.tensor(df[features].values.astype(np.float32))
    feature_tensor = torch.nan_to_num(feature_tensor, nan=0.0)
    target_tensor = torch.tensor(df[label].values.astype(np.float32)).to(
        torch.float32
    )
    dataset = TensorDataset(feature_tensor, target_tensor)
    train_dataset,test_dataset=random_split(dataset, get_train_test_split_size(len(dataset)))

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=6, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=6, shuffle=True)

    model = LinearNet(num_features=len(features)).to(get_device())
    loss_fn = nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    def train():
        model.train()
        for batch, (X, y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            output = model(X)
            loss = loss_fn(output.squeeze(), y)
            loss.backward()
            optimizer.step()
            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                size = len(train_dataset)
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

    def test():
        model.eval()
        with torch.no_grad():
            test_loss = 0.
            for batch, (X, y) in enumerate(test_dataloader):
                output = model(X)
                loss = loss_fn(output.squeeze(), y)
                test_loss += loss.item() * X.size(0)
            avg_loss = test_loss / len(test_dataset)

    num_epochs = 20
    for t in range(num_epochs):
        print(f"epoch {t}")
        train()
    test()
    return model.state_dict()


if __name__ == "__main__":
    torch.set_printoptions(sci_mode=False)
    path_to_dataframe_file = gan_mn_dir.parent / gan_mn_dataframe_filename
    state_dict = train_model(path_to_dataframe_file)
    print(f"saving state dict to {path_to_dataframe_file.parent}")
    torch.save(state_dict, path_to_dataframe_file.parent / "model.pth")
