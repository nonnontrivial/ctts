# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy~=1.26.4",
#     "pandas==2.1.4",
#     "torch~=2.2.2",
# ]
# ///

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# we need the package containing the model; add it to the path
try:
    model_path = list(Path.cwd().parent.rglob("model.py"))[0]
    model_parent = model_path.parent.as_posix()
    sys.path.append(model_parent)
except Exception as e:
    log.error(f"failed to add package to path {e}")
    sys.exit(1)
else:
    from model import NN

    log.info("loaded neural net")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

epochs = 150
features = [
    "Latitude",
    "Longitude",
    "Elevation",
    "CloudCover",
    "UTTimeHour",
    "MoonAlt",
    "MoonAz",
]


def get_datasets(data_dir_path: Path) -> tuple:
    df = pd.read_csv(data_dir_path / "gan.csv")
    feature_tensor = torch.tensor(df[features].values.astype(np.float32))
    feature_tensor = torch.nan_to_num(feature_tensor, nan=0.0)
    target_tensor = torch.tensor(df["SQMReading"].values.astype(np.float32)).to(
        torch.float32
    )

    data_tensor = TensorDataset(feature_tensor, target_tensor)
    train_size = int(0.8 * len(data_tensor))
    test_size = len(data_tensor) - train_size
    train_tensor, test_tensor = random_split(data_tensor, [train_size, test_size])

    train_dataloader = DataLoader(dataset=train_tensor, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(dataset=test_tensor, batch_size=16, shuffle=True)
    return train_dataloader, test_dataloader


def get_model():
    model = NN().to(device)
    return model


def train_model(
    data_loader: DataLoader,
    model: NN,
    loss_fn: nn.HuberLoss,
    optimizer: torch.optim.Adam,
) -> None:
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output.squeeze(), y)
        loss.backward()
        nn.utils.clip_grad.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            log.info(f"loss: {loss:>7f} [{current:>5d}]")


def evaluate_model(data_loader: DataLoader, model: NN, loss_fn: nn.HuberLoss) -> None:
    size = len(data_loader.dataset)
    test_loss = 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred.squeeze(), y)
            test_loss += loss.item() * X.size(0)
    test_loss /= size
    log.info(f"avg loss during eval was {test_loss:>8f}")


def main() -> None:
    torch.set_printoptions(sci_mode=False)

    data_dir_path = Path("./gan-data")
    assert data_dir_path.exists(), "data dir path does not exist!"

    train_dataloader, test_dataloader = get_datasets(data_dir_path)
    assert train_dataloader is not None, "no training data!"
    assert test_dataloader is not None, "no testing data!"

    model = get_model()

    loss_fn = nn.HuberLoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for i in range(epochs):
        log.info(f"epoch {i + 1}/{epochs}")
        train_model(train_dataloader, model, loss_fn, optimizer)

    model.eval()
    evaluate_model(test_dataloader, model, loss_fn)

    state_dict_filename = "model.pth"
    log.info(f"writing {state_dict_filename} to {model_path.parent}")
    torch.save(model.state_dict(), model_path.parent / state_dict_filename)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.warning("exiting")
