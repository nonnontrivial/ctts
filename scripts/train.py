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
import typing
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

epochs = 100
csv_filename = "gan.csv"
model_filename = "model.pth"
service_dir_path = list(Path.cwd().parent.rglob("service"))[0]

features = [
    "Latitude",
    "Longitude",
    "Elevation",
    "CloudCover",
    "UTTimeHour",
    "MoonAlt",
    "MoonAz",
]

# we need the package containing nn module; add it to the path
try:
    sys.path.append((list(Path.cwd().parent.rglob("neural_net"))[0]).as_posix())
except Exception as e:
    log.error(f"failed to add nn module to path {e}")
    sys.exit(1)
else:
    from nn import NN

    log.info("loaded neural net")


def get_datasets(data_dir_path: Path):
    """get the training and test data objects from the gan csv"""
    import numpy as np

    df = pd.read_csv(data_dir_path / csv_filename)

    torch.set_printoptions(sci_mode=False)
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


def get_model() -> typing.Any:
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = NN().to(device)
    return model


def train_model(
    data_loader: DataLoader,
    model: NN,
    loss_fn: nn.HuberLoss,
    optimizer: torch.optim.Adam,
):
    model.train()
    for batch, (X, y) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output.squeeze(), y)
        loss.backward()
        nn.utils.clip_grad.clip_grad_norm(model.parameters(), max_norm=5)
        optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            log.info(f"loss: {loss:>7f} [{current:>5d}]")


def evaluate_model(data_loader: DataLoader, model: NN, loss_fn: nn.HuberLoss):
    model.eval()
    with torch.no_grad():
        test_loss = 0
        for batch, (X, y) in enumerate(data_loader):
            pred = model(X)
            loss = loss_fn(pred.squeeze(), y)
            test_loss += loss.item() * X.size(0)


def main() -> None:
    data_dir_path = list(Path.cwd().parent.rglob("brightness_data"))[0]
    assert data_dir_path.exists(), "data dir path des not exist!"

    log.info("getting datasets from source file")
    train_dataloader, test_dataloader = get_datasets(data_dir_path)
    assert train_dataloader is not None, "no training data!"
    assert test_dataloader is not None, "no testing data!"

    model = get_model()

    loss_fn = nn.HuberLoss()
    learning_rate = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    log.info("training model")
    for epoch in range(epochs):
        log.info(f"epoch {epoch + 1}/{epochs}")
        train_model(train_dataloader, model, loss_fn, optimizer)

    log.info("running model on test data")
    evaluate_model(test_dataloader, model, loss_fn)

    log.info(f"writing {model_filename} to {service_dir_path}")
    torch.save(model.state_dict(), service_dir_path / model_filename)


if __name__ == "__main__":
    main()
